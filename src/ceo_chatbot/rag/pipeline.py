"""
 #! src/ceo_chatbot/rag/pipeline.py
 summary:
    RAG orchestration service for conversational retrieval-augmented generation.

 details:
    This module implements RagService class that manages document retrieval,
    prompt construction with conversation history, and LLM-based answer generation.
    Supports both single-turn and multi-turn conversations with automatic context management.
"""

from typing import Callable, Dict, Generator, Iterable, List, Tuple, Optional
from pathlib import Path
import logging
from langchain_core.documents import Document as LangchainDocument
from transformers import PreTrainedTokenizerBase
from ceo_chatbot.config import load_rag_config, load_prompt_template
from ceo_chatbot.rag.llm import get_reader_llm, LLMProvider
from ceo_chatbot.rag.retriever import load_faiss_index, get_retriever
from ceo_chatbot.rag.interpreter import QuestionInterpreter, Interpretation


class RagService:
    def __init__(
            self,
            config_path: str | Path = "conf/base/rag_config.yml",
            retriever: Callable[[str, int], List[LangchainDocument]] | None = None,
            skip_interpretation: bool = False
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
            skip_interpretation (bool): If True, skips question interpretation for testing (uses raw questions). Defaults to False.

        Returns:
            None

        Notes:
            Requires Hugging Face model configs; may be resource-intensive for large models.
            Use retriever injection for unit testing without FAISS index loading.
            skip_interpretation is intended for testing scenarios only.
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
        self.llm: LLMProvider = reader_llm
        self.tokenizer: Optional[PreTrainedTokenizerBase] = tokenizer

        # Get model's max input tokens and set context limit
        self.max_input_tokens = self.llm.max_context_tokens
        
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

        # Store testing override settings
        self.skip_interpretation = skip_interpretation

        # Initialize question interpreter for two-phase reasoning
        self.interpreter = QuestionInterpreter(self.llm)

    def _count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text using the LLM provider.

        details:
            Uses the provider's count_tokens method.
            Essential for estimating prompt lengths to prevent exceeding model context limits.

        Args:
            text (str): The text string to tokenize and count.

        Returns:
            int: Number of tokens in the text.
        """
        return self.llm.count_tokens(text)

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Applies chat template to messages.
        """
        if self.tokenizer:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for providers without a HF tokenizer (like Gemini)
            prompt = ""
            for msg in messages:
                role = msg["role"].upper()
                content = msg["content"]
                prompt += f"{role}: {content}\n\n"
            prompt += "ASSISTANT: "
            return prompt

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
        candidate_prompt = self._apply_template(messages)
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
            candidate_prompt = self._apply_template(messages)
            token_count = self._count_tokens(candidate_prompt)

        return truncated_history

    def _build_context(self, docs: List[LangchainDocument]) -> str:
        """
        Builds formatted context string from retrieved documents.

        details:
            Formats document content with numbered prefixes, line breaks,
            source titles, and source URLs for inclusion in the user message 
            during prompt construction.

        Args:
            docs (List[LangchainDocument]): List of retrieved documents with page_content and metadata.

        Returns:
            str: Formatted context string with document titles and source URLs.
        """
        context = "\nExtracted documents:\n"
        for i, doc in enumerate(docs):
            url = doc.metadata.get("url", "No URL available")
            title = doc.metadata.get("main_document_title") or doc.metadata.get("title") or f"Document {i}"
            context += f"--- SOURCE TITLE: {title} | URL: {url} ---\n{doc.page_content}\n\n"
        return context

    def _interpret_question_phase1(
        self,
        question: str,
        history: List[Dict[str, str]],
        skip_interpretation: bool
    ) -> Interpretation:
        """
        Phase 1: Interpret question intent and determine response strategy.
        
        details:
            Consolidates the interpretation logic that was duplicated between
            answer() and stream_answer() methods. Uses the question interpreter
            to analyze intent and determine optimal response strategy.

        Args:
            question (str): The current user question string to interpret.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.
            skip_interpretation (bool): If True, skips interpretation for testing (uses raw question).

        Returns:
            Interpretation: Structured analysis of question intent and strategy.
        """
        # Phase 1: Interpret question and determine strategy (unless skipped for testing)
        if skip_interpretation:
            # For testing: skip interpretation and use raw question with retrieval
            interpretation = Interpretation(
                original_question=question,
                interpreted_question=question,
                interpreted_successfully=False,  # Not interpreted
                interpretation_confidence=0.0,   # No interpretation performed
                routing_confidence=1.0,          # High confidence in default routing
                needs_retrieval=True,            # Default to retrieval for testing
                search_query=question,
                context_used="standalone",
                reasoning="Interpretation skipped - will answer raw question with retrieval"
            )
        else:
            # Normal interpretation
            interpretation = self.interpreter.interpret_question(question, history)

        return interpretation

    def _build_prompt_and_retrieve(
        self,
        interpretation: Interpretation,
        history: List[Dict[str, str]],
        num_retrieved_docs: int,
        num_docs_final: int
    ) -> Tuple[str, List[LangchainDocument], str, List[Dict[str, str]]]:
        """
        Build prompt and retrieve documents for retrieval-based responses.
        
        details:
            Consolidates the prompt building and retrieval logic that was duplicated
            across all four helper methods (_answer_with_retrieval, _stream_with_retrieval,
            _answer_from_history, _stream_from_history). This function handles document
            retrieval, context building, and prompt construction.

        Args:
            interpretation (Interpretation): Question interpretation results containing search query.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.
            num_retrieved_docs (int): Number of documents to retrieve initially from vector store.
            num_docs_final (int): Number of top documents to include in the final context.

        Returns:
            Tuple[str, List[LangchainDocument], str, List[Dict[str, str]]]: 
                - prompt: The formatted prompt string
                - docs: Retrieved documents
                - context: Built context string  
                - truncated_history: History after truncation
        """
        # Use interpreted search query for better retrieval
        search_query = interpretation.search_query or interpretation.interpreted_question

        # 1. Retrieve
        docs = self.retriever(search_query, k=num_retrieved_docs)
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
        truncated_history = self._truncate_history(history, context, search_query, base_system)

        # 4. Build conversational message list
        # Use multiturn user template if history exists, otherwise singleturn
        if truncated_history:
            user_content = self.multiturn_messages[1]["content"].format(context=context, question=search_query)
        else:
            user_content = self.singleturn_messages[1]["content"].format(context=context, question=search_query)
        messages = [base_system] + truncated_history + [
            {"role": "user", "content": user_content}
        ]

        # 5. Apply chat template to get final prompt
        prompt = self._apply_template(messages)
        return prompt, docs, context, truncated_history

    def _build_prompt_from_history(
        self,
        interpretation: Interpretation,
        history: List[Dict[str, str]]
    ) -> str:
        """
        Build prompt using conversation history and prompts.yml for history-based responses.
        
        details:
            Creates a prompt for history-based generation without document retrieval.
            Uses the interpreted question and conversation history to construct
            the appropriate prompt structure.

        Args:
            interpretation (Interpretation): Question interpretation results containing interpreted question.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.

        Returns:
            str: Formatted prompt string for history-based generation.
        """
        # Use conversation history and prompts.yml for answer generation
        # Pass empty docs list to _build_prompt to avoid retrieval
        prompt, docs, context, truncated_history = self._build_prompt(
            interpretation.interpreted_question, history, 0, 0
        )

        return prompt

    def _standard_generation(
        self,
        prompt: str
    ) -> str:
        """
        Generate answer using standard (non-streaming) LLM generation.
        
        details:
            Handles standard generation for the answer() method. This is the
            only difference between answer() and stream_answer() - the generation
            mechanism used.

        Args:
            prompt (str): The formatted prompt string to generate from.

        Returns:
            str: Generated answer string.
        """
        return self.llm.generate(prompt)

    def _stream_generation(
        self,
        prompt: str,
        max_new_tokens: int
    ) -> Iterable[str]:
        """
        Generate answer using streaming LLM generation.
        
        details:
            Handles streaming generation for the stream_answer() method. This is the
            only difference between answer() and stream_answer() - the generation
            mechanism used.

        Args:
            prompt (str): The formatted prompt string to generate from.
            max_new_tokens (int): Maximum number of tokens to generate in the response.

        Returns:
            Iterable[str]: Generator yielding answer chunks.
        """
        return self.llm.stream(prompt, max_new_tokens=max_new_tokens)

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
        prompt = self._apply_template(messages)
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
        interpretation = self._interpret_question_phase1(question, history, self.skip_interpretation)

        if debug:
            logging.info(f"[RAG DEBUG] Question: {question}")
            logging.info(f"[RAG DEBUG] Interpretation: {interpretation.reasoning}")
            logging.info(f"[RAG DEBUG] Needs retrieval: {interpretation.needs_retrieval}")
            logging.info(f"[RAG DEBUG] Interpretation confidence: {interpretation.interpretation_confidence}, Routing confidence: {interpretation.routing_confidence}")

        # Phase 2: Route to appropriate response strategy
        if interpretation.needs_retrieval:
            # Route 1: Retrieval-based response
            prompt, docs, context, truncated_history = self._build_prompt_and_retrieve(
                interpretation, history, num_retrieved_docs, num_docs_final
            )
            
            if debug:
                logging.info(f"[RAG DEBUG] Search query: {interpretation.search_query or interpretation.interpreted_question}")
                logging.info(f"[RAG DEBUG] Retrieved docs: {len(docs)}")
                logging.info(f"[RAG DEBUG] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

            # Generate answer using standard generation
            answer = self._standard_generation(prompt)

            if debug:
                logging.info(f"[RAG DEBUG] Answer length: {len(answer)} chars")
                logging.info(f"[RAG DEBUG] Answer preview: {answer[:100]}...")

            return answer, docs
        else:
            # Route 2: History-based response
            if debug:
                logging.info(f"[RAG DEBUG] Using conversation history for answer generation")

            # Build prompt using history
            prompt = self._build_prompt_from_history(interpretation, history)
            
            if debug:
                logging.info(f"[RAG DEBUG] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

            # Generate answer using standard generation
            answer = self._standard_generation(prompt)

            # Return empty docs list since no retrieval was performed
            return answer, []
    
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
        interpretation = self._interpret_question_phase1(question, history, self.skip_interpretation)

        if debug:
            logging.info(f"[RAG DEBUG STREAM] Question: {question}")
            logging.info(f"[RAG DEBUG STREAM] Interpretation: {interpretation.reasoning}")
            logging.info(f"[RAG DEBUG STREAM] Needs retrieval: {interpretation.needs_retrieval}")
            logging.info(f"[RAG DEBUG STREAM] Interpretation confidence: {interpretation.interpretation_confidence}, Routing confidence: {interpretation.routing_confidence}")

        # Phase 2: Route to appropriate streaming response strategy
        if interpretation.needs_retrieval:
            # Route 1: Retrieval-based streaming response
            prompt, docs, context, truncated_history = self._build_prompt_and_retrieve(
                interpretation, history, num_retrieved_docs, num_docs_final
            )
            
            if debug:
                logging.info(f"[RAG DEBUG STREAM] Search query: {interpretation.search_query or interpretation.interpreted_question}")
                logging.info(f"[RAG DEBUG STREAM] Retrieved docs: {len(docs)}")
                logging.info(f"[RAG DEBUG STREAM] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

            # Generate answer using streaming generation
            answer_generator = self._stream_generation(prompt, max_new_tokens)

            return answer_generator, docs
        else:
            # Route 2: History-based streaming response
            if debug:
                logging.info(f"[RAG DEBUG STREAM] Using conversation history for answer generation")

            # Build prompt using history
            prompt = self._build_prompt_from_history(interpretation, history)
            
            if debug:
                logging.info(f"[RAG DEBUG STREAM] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

            # Generate answer using streaming generation
            answer_generator = self._stream_generation(prompt, max_new_tokens)

            # Return empty docs list since no retrieval was performed
            return answer_generator, []
