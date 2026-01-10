"""
 #! src/ceo_chatbot/rag/pipeline.py
 summary:
    RAG orchestration service for conversational retrieval-augmented generation.

 details:
    This module implements RagService class that manages document retrieval,
    prompt construction with conversation history, and LLM-based answer generation.
    Supports both single-turn and multi-turn conversations with automatic context management.
"""

from typing import Callable, Dict, Generator, Iterable, List, Tuple
from pathlib import Path
from threading import Thread
import logging
from langchain_core.documents import Document as LangchainDocument
from transformers import AutoConfig, PreTrainedTokenizerBase, TextIteratorStreamer
from ceo_chatbot.config import load_rag_config, load_prompt_template
from ceo_chatbot.rag.llm import get_reader_llm
from ceo_chatbot.rag.retriever import load_faiss_index, get_retriever
from ceo_chatbot.rag.interpreter import QuestionInterpreter, Interpretation


class RagService:
    def __init__(
            self,
            config_path: str | Path = "conf/base/rag_config.yml",
            retriever: Callable[[str, int], List[LangchainDocument]] | None = None
    ) -> None:
        """
        Initializes the RagService with configuration, model, and retriever setup.

        details:
            Loads RAG configuration, initializes the language model and tokenizer,
            sets dynamic context limits, and prepares for conversation processing.
            Supports dependency injection of retriever for testing purposes.

        Args:
            config_path (str | Path): Path to the RAG configuration YAML file containing model and data paths.
            retriever (Callable[[str, int], List[LangchainDocument]] | None): Optional retriever function for dependency injection in tests.

        Returns:
            None

        Notes:
            Requires Hugging Face model configs; may be resource-intensive for large models.
            Use retriever injection for unit testing without FAISS index loading.
        """
        
        # Load config
        self.config = load_rag_config(config_path)

        # Load or inject retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            faiss_index = load_faiss_index(
                index_dir=self.config.vectorstore_path,
                embedding_model_name=self.config.embedding_model_name,
            )
            self.retriever = get_retriever(faiss_index)

        # Load LLM & tokenizer
        reader_llm, tokenizer = get_reader_llm(
            model_name=self.config.reader_model_name,
        )
        self.llm = reader_llm
        self.tokenizer: PreTrainedTokenizerBase = tokenizer

        # Get model's max input tokens and set context limit
        model_config = AutoConfig.from_pretrained(self.config.reader_model_name)
        self.max_input_tokens = model_config.max_position_embeddings
        
        # Reserve tokens for generation response
        reserve_for_generation = 500
        self.max_context_tokens = self.max_input_tokens - reserve_for_generation

        # Load base instructions and prompt extensions
        self.base_instructions = load_prompt_template(
            prompt_file=self.config.prompt_file, prompt_key='base_instructions'
        )
        self.singleturn_messages = load_prompt_template(
            prompt_file=self.config.prompt_file, prompt_key='singleturn_prompt'
        )
        self.multiturn_messages = load_prompt_template(
            prompt_file=self.config.prompt_file, prompt_key='multiturn_prompt'
        )

        # Initialize question interpreter for two-phase reasoning
        self.interpreter = QuestionInterpreter(self.llm, self.tokenizer)

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text using the tokenizer.

        details:
            Uses the tokenizer's encode method to tokenize the text and count tokens.
            Essential for estimating prompt lengths to prevent exceeding model context limits.

        Args:
            text (str): The text string to tokenize and count.

        Returns:
            int: Number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def _truncate_history(
        self,
        history: List[Dict[str, str]],
        context: str,
        question: str,
        base_system_message: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """
        Truncates conversation history if the combined prompt would exceed the context limit.

        details:
            Builds a candidate prompt including system message, history, and current question.
            Calculates token count and removes oldest history messages iteratively until
            the total prompt fits within the model's maximum context tokens minus generation reserve.

        Args:
            history (List[Dict[str, str]]): List of previous message dictionaries with 'role' and 'content'.
            context (str): The retrieved document context string.
            question (str): The current user question string.
            base_system_message (Dict[str, str]): The system message dict to use (must have 'role' and 'content').

        Returns:
            List[Dict[str, str]]: The truncated history list, containing only messages that fit within context limits.
        """
        # Build the current user message content with retrieved context
        # Use multiturn user template if history exists, otherwise singleturn
        if history:
            user_content = self.multiturn_messages[1]["content"].format(context=context, question=question)
        else:
            user_content = self.singleturn_messages[1]["content"].format(context=context, question=question)

        # Construct full message list for candidate prompt
        messages = [base_system_message] + history + [
            {"role": "user", "content": user_content}
        ]

        # Build candidate prompt and check token count
        candidate_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_count = self._count_tokens(candidate_prompt)

        if token_count <= self.max_context_tokens:
            return history

        # Remove oldest history messages until prompt fits
        truncated_history = history[:]
        while truncated_history and token_count > self.max_context_tokens:
            truncated_history.pop(0)  # Remove oldest message
            messages = [base_system_message] + truncated_history + [
                {"role": "user", "content": user_content}
            ]
            candidate_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_count = self._count_tokens(candidate_prompt)

        return truncated_history

    def _build_context(self, docs: List[LangchainDocument]) -> str:
        """
        Builds formatted context string from retrieved documents.

        details:
            Formats document content with numbered prefixes and line breaks
            for inclusion in the user message during prompt construction.

        Args:
            docs (List[LangchainDocument]): List of retrieved documents with page_content.

        Returns:
            str: Formatted context string with document separations.
        """
        context = "\nExtracted documents:\n"
        context += "".join(
            [
                f"Document {i}:::\n{doc.page_content}\n"
                for i, doc in enumerate(docs)
            ]
        )
        return context

    def _build_prompt(self,
                      question: str,
                      history: List[Dict[str, str]] = [],
                      num_retrieved_docs: int = 30,
                      num_docs_final: int = 5
                      ) -> Tuple[str, List[LangchainDocument], str, List[Dict[str, str]]]:

        # 1. Retrieve
        docs = self.retriever(question, k=num_retrieved_docs)
        docs = docs[:num_docs_final]

        # 2. Build context
        context = self._build_context(docs)

        # Construct system content from base instructions + appropriate extensions
        base_content = self.base_instructions[0]["content"]  # Extract content from message
        if history:
            # Add multiturn conversation extensions
            system_content = base_content + "\n\n" + self.multiturn_messages[0]["content"]
        else:
            # Add single-turn specific extensions
            system_content = base_content + "\n\n" + self.singleturn_messages[0]["content"]
        base_system = {"role": "system", "content": system_content}

        # 3. Truncate history if necessary to fit context limit
        truncated_history = self._truncate_history(history, context, question, base_system)

        # 4. Build conversational message list
        # Use multiturn user template if history exists, otherwise singleturn
        if truncated_history:
            user_content = self.multiturn_messages[1]["content"].format(context=context, question=question)
        else:
            user_content = self.singleturn_messages[1]["content"].format(context=context, question=question)
        messages = [base_system] + truncated_history + [
            {"role": "user", "content": user_content}
        ]

        # 5. Apply chat template to get final prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, docs, context, truncated_history
         
    
    def answer(
        self,
        question: str,
        history: List[Dict[str, str]] = [],
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        debug: bool = False
    ) -> Tuple[str, List[LangchainDocument]]:
        """
        Performs two-phase reasoning for conversational RAG.

        Phase 1: Interpret question intent and determine response strategy
        Phase 2: Route to appropriate handler (direct answer vs retrieval + generation)

        details:
            Uses intelligent routing to balance conversational understanding with factual accuracy.
            For conversational follow-ups, provides direct answers from history.
            For knowledge-seeking questions, retrieves relevant documents and generates comprehensive responses.

        Args:
            question (str): The current user question string to answer.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.
            num_retrieved_docs (int): Number of documents to retrieve initially from vector store.
            num_docs_final (int): Number of top documents to include in the final context.
            debug (bool): Enable debug logging of internal pipeline steps if True.

        Returns:
            Tuple[str, List[LangchainDocument]]: Generated answer string and list of retrieved documents.
        """
        # Phase 1: Interpret question and determine strategy
        interpretation = self.interpreter.interpret_question(question, history)

        if debug:
            logging.info(f"[RAG DEBUG] Question: {question}")
            logging.info(f"[RAG DEBUG] Interpretation: {interpretation.reasoning}")
            logging.info(f"[RAG DEBUG] Needs retrieval: {interpretation.needs_retrieval}")
            logging.info(f"[RAG DEBUG] Confidence: {interpretation.confidence}")

        # Phase 2: Route to appropriate response strategy
        if interpretation.needs_retrieval:
            return self._answer_with_retrieval(
                interpretation, history, num_retrieved_docs, num_docs_final, debug
            )
        else:
            return self._answer_from_history(interpretation, debug)
    
    def stream_answer(
        self,
        question: str,
        history: List[Dict[str, str]] = [],
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        max_new_tokens: int = 500,
        debug: bool = False
    ) -> Tuple[Iterable[str], List[LangchainDocument]]:
        """
        Streaming version of answer() with two-phase reasoning support.

        Phase 1: Interpret question intent and determine response strategy
        Phase 2: Route to appropriate streaming handler

        details:
            Uses intelligent routing to balance conversational understanding with factual accuracy.
            For conversational follow-ups, streams direct answers from history.
            For knowledge-seeking questions, retrieves documents and streams comprehensive responses.

        Args:
            question (str): The current user question string to answer.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.
            num_retrieved_docs (int): Number of documents to retrieve initially from vector store.
            num_docs_final (int): Number of top documents to include in the final context.
            max_new_tokens (int): Maximum number of tokens to generate in the response.
            debug (bool): Enable debug logging of internal pipeline steps if True.

        Returns:
            Tuple[Iterable[str], List[LangchainDocument]]: Generator yielding answer chunks and list of retrieved documents.
        """
        # Phase 1: Interpret question and determine strategy
        interpretation = self.interpreter.interpret_question(question, history)

        if debug:
            logging.info(f"[RAG DEBUG STREAM] Question: {question}")
            logging.info(f"[RAG DEBUG STREAM] Interpretation: {interpretation.reasoning}")
            logging.info(f"[RAG DEBUG STREAM] Needs retrieval: {interpretation.needs_retrieval}")
            logging.info(f"[RAG DEBUG STREAM] Confidence: {interpretation.confidence}")

        # Phase 2: Route to appropriate streaming response strategy
        if interpretation.needs_retrieval:
            return self._stream_with_retrieval(
                interpretation, history, num_retrieved_docs, num_docs_final, max_new_tokens, debug
            )
        else:
            return self._stream_from_history(interpretation, debug)

    def _answer_with_retrieval(
        self,
        interpretation: Interpretation,
        history: List[Dict[str, str]],
        num_retrieved_docs: int,
        num_docs_final: int,
        debug: bool
    ) -> Tuple[str, List[LangchainDocument]]:
        """
        Handles questions that require document retrieval and generation.

        Args:
            interpretation: Question interpretation results
            history: Conversation history
            num_retrieved_docs: Number of docs to retrieve
            num_docs_final: Number of docs to include in context
            debug: Enable debug logging

        Returns:
            Tuple of (answer, retrieved_docs)
        """
        # Use interpreted search query for better retrieval
        search_query = interpretation.search_query or interpretation.interpreted_question

        # Build prompt with retrieved documents
        prompt, docs, context, truncated_history = self._build_prompt(
            search_query, history, num_retrieved_docs, num_docs_final
        )

        if debug:
            logging.info(f"[RAG DEBUG] Search query: {search_query}")
            logging.info(f"[RAG DEBUG] Retrieved docs: {len(docs)}")
            logging.info(f"[RAG DEBUG] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

        # Generate answer
        outputs = self.llm(prompt)
        answer = outputs[0]["generated_text"]

        if debug:
            logging.info(f"[RAG DEBUG] Answer length: {len(answer)} chars")
            logging.info(f"[RAG DEBUG] Answer preview: {answer[:100]}...")

        return answer, docs

    def _answer_from_history(
        self,
        interpretation: Interpretation,
        debug: bool
    ) -> Tuple[str, List[LangchainDocument]]:
        """
        Handles conversational follow-ups that can be answered directly from history.

        Args:
            interpretation: Question interpretation results
            debug: Enable debug logging

        Returns:
            Tuple of (direct_answer, empty_docs_list)
        """
        # Return the direct answer from interpretation
        answer = interpretation.direct_answer or interpretation.interpreted_question

        if debug:
            logging.info(f"[RAG DEBUG] Direct answer from history: {len(answer)} chars")
            logging.info(f"[RAG DEBUG] Answer preview: {answer[:100]}...")

        # Return empty docs list since no retrieval was performed
        return answer, []

    def _stream_with_retrieval(
        self,
        interpretation: Interpretation,
        history: List[Dict[str, str]],
        num_retrieved_docs: int,
        num_docs_final: int,
        max_new_tokens: int,
        debug: bool
    ) -> Tuple[Iterable[str], List[LangchainDocument]]:
        """
        Streams answers that require document retrieval and generation.

        Args:
            interpretation: Question interpretation results
            history: Conversation history
            num_retrieved_docs: Number of docs to retrieve
            num_docs_final: Number of docs to include in context
            max_new_tokens: Maximum tokens to generate
            debug: Enable debug logging

        Returns:
            Tuple of (answer_generator, retrieved_docs)
        """
        # Use interpreted search query for better retrieval
        search_query = interpretation.search_query or interpretation.interpreted_question

        # Build prompt with retrieved documents
        prompt, docs, context, truncated_history = self._build_prompt(
            search_query, history, num_retrieved_docs, num_docs_final
        )

        if debug:
            logging.info(f"[RAG DEBUG STREAM] Search query: {search_query}")
            logging.info(f"[RAG DEBUG STREAM] Retrieved docs: {len(docs)}")
            logging.info(f"[RAG DEBUG STREAM] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

        # Prepare streaming
        model = self.llm.model
        tokenizer = self.tokenizer

        # Tokenize input and move to model device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
        )

        # Run generate() in a background thread to iterate over streamer
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        def token_generator() -> Generator[str, None, None]:
            for text in streamer:
                # 'text' is already decoded text chunk
                yield text

        return token_generator(), docs

    def _stream_from_history(
        self,
        interpretation: Interpretation,
        debug: bool
    ) -> Tuple[Iterable[str], List[LangchainDocument]]:
        """
        Streams direct answers from conversation history.

        Args:
            interpretation: Question interpretation results
            debug: Enable debug logging

        Returns:
            Tuple of (answer_generator, empty_docs_list)
        """
        # Get the direct answer from interpretation
        answer = interpretation.direct_answer or interpretation.interpreted_question

        if debug:
            logging.info(f"[RAG DEBUG STREAM] Direct answer from history: {len(answer)} chars")
            logging.info(f"[RAG DEBUG STREAM] Answer preview: {answer[:100]}...")

        # Convert direct answer to streaming generator
        def token_generator() -> Generator[str, None, None]:
            # Yield the answer in chunks to simulate streaming
            # For simplicity, yield in reasonable-sized chunks
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                yield answer[i:i + chunk_size]

        # Return empty docs list since no retrieval was performed
        return token_generator(), []
