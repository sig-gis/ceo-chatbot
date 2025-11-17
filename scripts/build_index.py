from pathlib import Path
from ceo_chat.config import load_rag_config
from ceo_chatbot.ingest.index_builder import build_and_save_index

# Huggingface docs will need to be replaced with CEO docs
# and build_and_save index from ceo_chatbot.ingest.index_builder will need to be updated:
# raw_docs = <replace load_hf_docs() with load CEO docs logic>
DATASET_NAME = "m-ric/huggingface_doc"
SPLIT = "train"


def main(
        config_path: str | Path = "conf/base/rag_config.yml"
) -> None:
    config = load_rag_config(config_path)
    build_and_save_index(
        output_dir=Path(config.vectorstore_path),
        dataset_name=DATASET_NAME,
        split=SPLIT,
        embedding_model_name=config.embedding_model_name,
    )
    print(f"FAISS index saved to {config.vectorstore_path}")


if __name__ == "__main__":
    main()
