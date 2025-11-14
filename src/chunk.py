import tiktoken
from langchain_community.document_loaders import rst
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_rst(file_path):
    loader = rst.UnstructuredRSTLoader(file_path=file_path, mode="elements")
    docs = loader.load()
    return docs

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_semantically_with_size_constraint(
    docs: list[Document], 
    chunk_size: int = 256, 
    chunk_overlap: int = 25
) -> list[Document]:
    """
    Groups documents by titles, then splits large chunks (exceeding chunk_size) by token size.
    """
    final_chunks = []
    current_chunk_elements = []
    main_document_title = ""

    # Find the main document title
    for doc in docs:
        if doc.metadata.get("category") == "Title" and doc.metadata.get("category_depth") == 0:
            main_document_title = doc.page_content.strip()
            break
    
    content_docs = [doc for doc in docs if not (doc.metadata.get("category") == "Title" and doc.metadata.get("category_depth") == 0)]

    # Helper to process and split a semantic chunk
    def process_semantic_chunk(elements):
        if not elements:
            return

        combined_content = "\n\n".join([el.page_content for el in elements])
        
        # Check if the semantic chunk is larger than the target size
        token_count = num_tokens_from_string(combined_content)
        
        base_metadata = elements[0].metadata.copy()
        if main_document_title:
            base_metadata["main_document_title"] = main_document_title

        if token_count > chunk_size:
            # This chunk is too big, split it recursively
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            sub_chunks = text_splitter.create_documents([combined_content], metadatas=[base_metadata])
            final_chunks.extend(sub_chunks)
        else:
            # This chunk is small enough, add it as is
            new_chunk = Document(
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


if __name__=="__main__":
    # docs = load_rst('./data/index.rst')
    docs = load_rst('./data/preparing.rst')
    # docs = load_rst('./data/start.rst')

    print(f"Loaded {len(docs)} elements from the RST file.")
    
    print("\n" + "="*50)
    print("STRATEGY 2: HYBRID CHUNKING (SEMANTIC + Recursive Splitting over token limit)")
    print("="*50)

    target_chunk_size = 256
    hybrid_chunks = chunk_semantically_with_size_constraint(docs, chunk_size=target_chunk_size)

    print(f"Created {len(hybrid_chunks)} hybrid chunks with a target size of {target_chunk_size} tokens.\n")

    for i, chunk in enumerate(hybrid_chunks):
        chunk_token_count = num_tokens_from_string(chunk.page_content)
        print(f"----- CHUNK {i+1} (Tokens: {chunk_token_count}) -----")
        print(f"METADATA: {chunk.metadata}")
        print(f"CONTENT:\n{chunk.page_content}")
        print("-" * 20 + "\n")
