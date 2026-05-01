import argparse
import logging
from pathlib import Path

import torch

from ceo_chatbot.config import AppSettings, load_rag_config
from ceo_chatbot.ingest.chunking import semantic_recursive_chunks
from ceo_chatbot.ingest.index_builder import build_faiss_index
from ceo_chatbot.ingest.loaders import load_rst_docs
from ceo_chatbot.storage import GCSStorage


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from docs and upload result to GCS.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for the embedding model (default: auto-detect)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = _resolve_device(args.device)
    logging.info("Using device: %s", device)

    config = load_rag_config()
    settings = AppSettings()

    # docs_path is set in conf/base/rag_config.yml under embeddings.docs_path
    docs_dir = Path(config.docs_path)

    # Use local docs if they exist; otherwise download from GCS
    if any(docs_dir.rglob("*.rst")):
        logging.info("Local docs found at %s, skipping GCS download", docs_dir)
    else:
        logging.info("No local docs at %s, downloading from GCS...", docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        gcs_docs = GCSStorage(settings.docs_bucket_name)
        n = gcs_docs.download_prefix(settings.folder_prefix, docs_dir)
        logging.info("Downloaded %d files from gs://%s/%s", n, settings.docs_bucket_name, settings.folder_prefix)

    # Load, chunk, embed, build FAISS index
    raw_docs = load_rst_docs(str(docs_dir))
    chunks = semantic_recursive_chunks(raw_docs, chunk_size=config.chunk_size)
    index = build_faiss_index(chunks, device=device)

    # vectorstore_path is set in conf/base/rag_config.yml under embeddings.vectorstore_path
    index_dir = Path(config.vectorstore_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))
    logging.info("Index saved to %s", index_dir)

    # Upload index.faiss and index.pkl to GCS
    gcs_db = GCSStorage(settings.db_bucket_name)
    prefix = str(config.vectorstore_gcs)
    for fname in ("index.faiss", "index.pkl"):
        gcs_db.upload(index_dir / fname, f"{prefix}/{fname}")

    gcs_path = f"gs://{settings.db_bucket_name}/{prefix}/"
    logging.info("Index uploaded to %s", gcs_path)
    print(gcs_path)


if __name__ == "__main__":
    main()
