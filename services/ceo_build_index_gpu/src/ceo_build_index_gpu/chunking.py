from pathlib import Path
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from ceo_build_index_gpu.config import load_rag_config


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

    def process_semantic_chunk(elements):
        if not elements:
            return

        combined_content = "\n\n".join([el.page_content for el in elements])
        token_count = num_tokens_from_string_using_model(combined_content, tokenizer)

        base_metadata = elements[0].metadata.copy()
        if main_document_title:
            base_metadata["main_document_title"] = main_document_title

        if token_count > chunk_size:
            sub_chunks = text_splitter.create_documents([combined_content], metadatas=[base_metadata])
            final_chunks.extend(sub_chunks)
        else:
            new_chunk = LangchainDocument(
                page_content=combined_content,
                metadata=base_metadata,
            )
            final_chunks.append(new_chunk)

    # Iterate through docs to create semantic chunks.
    # Depth-0 titles mark a new RST file: flush the current chunk with the old title,
    # then update main_document_title for all subsequent chunks.
    for doc in knowledge_base:
        if doc.metadata.get("category") == "Title" and doc.metadata.get("category_depth") == 0:
            process_semantic_chunk(current_chunk_elements)
            main_document_title = doc.page_content.strip()
            current_chunk_elements = []
        else:
            current_chunk_elements.append(doc)

    process_semantic_chunk(current_chunk_elements)

    return final_chunks
