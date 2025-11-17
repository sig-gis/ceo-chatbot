from typing import Generator, Iterable, List, Tuple
from pathlib import Path
from threading import Thread
from langchain_core.documents import Document as LangchainDocument
from transformers import PreTrainedTokenizerBase, TextIteratorStreamer
from ceo_chatbot.config import load_rag_config, load_prompt_template
from ceo_chatbot.rag.llm import get_reader_llm
from ceo_chatbot.rag.retriever import load_faiss_index, get_retriever


class RagService:
    def __init__(
            self, 
            config_path: str | Path = "conf/base/rag_config.yml"
    ) -> None:
        """
        RAG orchestration service.
        """
        
        # Load config
        self.config = load_rag_config(config_path)

        # Load retriever
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

        # Build prompt template -- only one prompt for now
        chat_template = load_prompt_template(
            prompt_file=self.config.prompt_file
        )
        self.prompt_template = self.tokenizer.apply_chat_template(
            chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_context(self, docs: List[LangchainDocument]) -> str:
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
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[str, List[LangchainDocument]]:
        """
        Full RAG flow:
          - retrieve docs
          - build prompt with context
          - call LLM
          - return answer + docs
        """
        # 1. Retrieve
        docs = self.retriever(question, k=num_retrieved_docs)
        docs = docs[:num_docs_final]

        # 2. Build prompt
        context = self._build_context(docs)
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        # 3. Generate answer
        outputs = self.llm(prompt)
        answer = outputs[0]["generated_text"]

        return answer, docs
    
    def stream_answer(
        self,
        question: str,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
        max_new_tokens: int = 500,
    ) -> Tuple[Iterable[str], List[LangchainDocument]]:
        """
        Streaming version of answer():
          - returns a generator
          - generator yields text chunks as they are produced by the model
          - this can be demo'ed in a UI, e.g. streamlit: st.write_stream
        """
        # 1. Retrieve docs
        docs = self.retriever(question, k=num_retrieved_docs)
        docs = docs[:num_docs_final]

        # 2. Build prompt
        context = self._build_context(docs)
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        # 3. Prepare streaming
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
