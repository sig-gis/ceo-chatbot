from pathlib import Path

from ceo_chatbot.config import load_rag_config
from ceo_chatbot.ingest.index_builder import build_and_save_index
from ceo_chatbot.ingest.loaders import sync_ceo_docs

def main(
        config_path: str | Path = "conf/base/rag_config.yml"
) -> None:
    config = load_rag_config(config_path)
    
    ceo_docs = sync_ceo_docs()
    
    build_and_save_index(
        output_dir=Path(config.vectorstore_path),
    )
    print(f"FAISS index saved to {config.vectorstore_path}")


if __name__ == "__main__":
    main()
