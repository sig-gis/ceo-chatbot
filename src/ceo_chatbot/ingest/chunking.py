from typing import List
from pathlib import Path
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from ceo_chatbot.config import load_rag_config


def split_documents(
    knowledge_base: List[LangchainDocument],
    chunk_size: int = 512,
    config_path: str | Path = "conf/base/rag_config.yml"
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    Removes duplicate chunks based on text content.
    """
    MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
    ]
    config = load_rag_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed: List[LangchainDocument] = []
    for doc in knowledge_base:
        docs_processed.extend(text_splitter.split_documents([doc]))

    # Remove duplicates by page_content
    seen = set()
    unique_docs: List[LangchainDocument] = []
    for doc in docs_processed:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs
