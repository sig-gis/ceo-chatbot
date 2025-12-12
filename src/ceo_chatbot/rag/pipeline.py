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

        # Load default and multiturn prompt messages
        self.default_messages = load_prompt_template(
            prompt_file=self.config.prompt_file, prompt_key='default_prompt'
        )
        self.multiturn_messages = load_prompt_template(
            prompt_file=self.config.prompt_file, prompt_key='multiturn_prompt'
        )

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
        user_content = (
            f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}"
        )

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

    def answer(
        self,
        question: str,
        history: List[Dict[str, str]] = [],
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        debug: bool = False
    ) -> Tuple[str, List[LangchainDocument]]:
        """
        Performs full RAG flow with conversation history support.

        details:
            Retrieves relevant documents using the configured retriever,
            builds prompt incorporating conversation history, truncates history if needed,
            generates response using the LLM, and returns answer with source documents.
            History truncation ensures prompt stays within model context limits.

        Args:
            question (str): The current user question string to answer.
            history (List[Dict[str, str]]): List of previous conversation messages with 'role' and 'content'.
            num_retrieved_docs (int): Number of documents to retrieve initially from vector store.
            num_docs_final (int): Number of top documents to include in the final context.
            debug (bool): Enable debug logging of internal pipeline steps if True.

        Returns:
            Tuple[str, List[LangchainDocument]]: Generated answer string and list of retrieved documents.
        """
        # 1. Retrieve
        docs = self.retriever(question, k=num_retrieved_docs)
        docs = docs[:num_docs_final]

        # 2. Build context
        context = self._build_context(docs)

        # Select appropriate prompt based on history presence
        base_messages = self.multiturn_messages if history else self.default_messages
        base_system = base_messages[0]

        # 3. Truncate history if necessary to fit context limit
        history = self._truncate_history(history, context, question, base_system)

        # 4. Build conversational message list
        user_content = (
            f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}"
        )
        messages = [base_system] + history + [
            {"role": "user", "content": user_content}
        ]

        # 5. Apply chat template to get final prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if debug:
            logging.info(f"[RAG DEBUG] Question: {question}")
            logging.info(f"[RAG DEBUG] History messages: {len(history)}")
            logging.info(f"[RAG DEBUG] Context docs: {len(docs)}")
            logging.info(f"[RAG DEBUG] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

        # 6. Generate answer
        outputs = self.llm(prompt)
        answer = outputs[0]["generated_text"]

        if debug:
            logging.info(f"[RAG DEBUG] Answer length: {len(answer)} chars")
            logging.info(f"[RAG DEBUG] Answer preview: {answer[:100]}...")

        return answer, docs
    
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
        Streaming version of answer() with conversation history support.

        details:
            Identical to answer() method but returns a generator that yields answer chunks
            as they are produced by the LLM, enabling real-time streaming for UI display.
            Maintains the same conversation context processing and history truncation.

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
        # 1. Retrieve docs
        docs = self.retriever(question, k=num_retrieved_docs)
        docs = docs[:num_docs_final]

        # 2. Build context
        context = self._build_context(docs)

        # Select appropriate prompt based on history presence
        base_messages = self.multiturn_messages if history else self.default_messages
        base_system = base_messages[0]

        # 3. Truncate history if necessary to fit context limit
        history = self._truncate_history(history, context, question, base_system)

        # 4. Build conversational message list
        user_content = (
            f"Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}"
        )
        messages = [base_system] + history + [
            {"role": "user", "content": user_content}
        ]

        # 5. Apply chat template to get final prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if debug:
            logging.info(f"[RAG DEBUG STREAM] Question: {question}")
            logging.info(f"[RAG DEBUG STREAM] History messages: {len(history)}")
            logging.info(f"[RAG DEBUG STREAM] Context docs: {len(docs)}")
            logging.info(f"[RAG DEBUG STREAM] Prompt tokens: {self._count_tokens(prompt)} / {self.max_context_tokens}")

        # 6. Prepare streaming
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
