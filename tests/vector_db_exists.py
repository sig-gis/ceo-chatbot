from ceo_chatbot.rag.retriever import load_faiss_index, get_retriever
from ceo_chatbot.ingest.embeddings import get_embedding_model
from ceo_chatbot.config import load_rag_config, load_prompt_template

config = load_rag_config("conf/base/rag_config.yml")


faiss_index = load_faiss_index(
                index_dir=config.vectorstore_path,
                embedding_model_name=config.embedding_model_name,
            )
retriever = get_retriever(faiss_index)

q = "What is Collect Earth Online?"
docs = retriever(q, k=20)
# docs = docs[:num_docs_final]
print(len(docs))
print(docs[:5])