from typing import List
from pathlib import Path
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from ceo_chatbot.config import load_rag_config


def _get_tokenizer(config_path: str | Path = "conf/base/rag_config.yml") -> AutoTokenizer:
    """Load the tokenizer used by the embedding model from the RAG config."""
    config = load_rag_config(config_path)
    return AutoTokenizer.from_pretrained(config.embedding_model_name)


def num_tokens_from_string_using_model(string: str, tokenizer: AutoTokenizer) -> int:
    """Returns the number of tokens in a text string using the specified tokenizer."""
    return len(tokenizer.encode(string))


def recursive_splitter(
    config_path: str | Path = "conf/base/rag_config.yml",
    chunk_size: int = 512
) -> TextSplitter:
    
    tokenizer = _get_tokenizer(config_path)

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

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return text_splitter

def semantic_recursive_chunks(
    knowledge_base: list[LangchainDocument],
    chunk_size: int = 512,
    config_path: str | Path = "conf/base/rag_config.yml"
) -> list[LangchainDocument]:
    """
    Groups documents by titles, then splits oversized chunks with a recursive text splitter.
    """
    final_chunks = []
    current_chunk_elements = []
    main_document_title = ""

    tokenizer = _get_tokenizer(config_path)
    text_splitter = recursive_splitter(config_path, chunk_size)

    # Find the main document title
    for doc in knowledge_base:
        if doc.metadata.get("category") == "Title" and doc.metadata.get("category_depth") == 0:
            main_document_title = doc.page_content.strip()
            break

    content_docs = [doc for doc in knowledge_base if not (doc.metadata.get("category") == "Title" and doc.metadata.get("category_depth") == 0)]

    # Helper to process and split a semantic chunk
    def process_semantic_chunk(elements):
        if not elements:
            return

        combined_content = "\n\n".join([el.page_content for el in elements])

        # Check if the semantic chunk is larger than the target size
        token_count = num_tokens_from_string_using_model(combined_content, tokenizer)
        
        base_metadata = elements[0].metadata.copy()
        if main_document_title:
            base_metadata["main_document_title"] = main_document_title

        if token_count > chunk_size:
            # This chunk is too big, split it recursively
            sub_chunks = text_splitter.create_documents([combined_content], metadatas=[base_metadata])
            final_chunks.extend(sub_chunks)
        else:
            # This chunk is small enough, add it as is
            new_chunk = LangchainDocument(
                page_content=combined_content,
                metadata=base_metadata,
            )
            final_chunks.append(new_chunk)

    # Iterate through docs to create semantic chunks
    for doc in content_docs:
        if doc.metadata.get("category") == "Title":
            process_semantic_chunk(current_chunk_elements)
            current_chunk_elements = [doc]
        else:
            current_chunk_elements.append(doc)
    
    # Process the last remaining chunk
    process_semantic_chunk(current_chunk_elements)

    return final_chunks

def split_documents(
    knowledge_base: List[LangchainDocument],
    chunk_size: int = 512,
    config_path: str | Path = "conf/base/rag_config.yml"
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    Removes duplicate chunks based on text content.
    """
    text_splitter = recursive_splitter(config_path, chunk_size)

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
