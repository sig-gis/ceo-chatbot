import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ceo_rag_chatbot.config import get_settings, load_chatbot_config
from ceo_rag_chatbot.rag.pipeline import RagService
from ceo_rag_chatbot.rag.retriever import get_retriever, load_faiss_index
from ceo_chatbot_core.storage import GCSStorage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download the FAISS index from GCS and initialise RagService on startup."""
    t0 = time.monotonic()
    settings = get_settings()
    config = load_chatbot_config()

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    # Download index.faiss and index.pkl from GCS into the local directory.
    # Cloud Run containers start fresh, so the files are never present on startup.
    # TODO: add smart-load (skip download if local copy is newer than GCS blob)
    #       using GCSStorage.blob_updated() once we want to support local dev speed.
    config.vectorstore_path.mkdir(parents=True, exist_ok=True)
    
    # Prefix comes from rag_config.yml (vectorstore_gcs) — same value the pipeline
    # used when uploading, so the download path always matches the upload path.
    index_prefix = str(config.vectorstore_gcs)
    gcs = GCSStorage(bucket_name=settings.db_bucket_name, project=settings.google_cloud_project)
    for fname in ("index.faiss", "index.pkl"):
        gcs.download(
            f"{index_prefix}/{fname}",
            config.vectorstore_path / fname,
        )
    logger.info("index downloaded to %s", config.vectorstore_path)

    # Build the retriever from the downloaded index. We inject it into RagService
    # so RagService loads from config.vectorstore_path rather than the
    # vectorstore_path in rag_config.yml — without needing to change pipeline.py.
    faiss_index = load_faiss_index(config.vectorstore_path, config.embedding_model_name)
    retriever = get_retriever(faiss_index)

    # RagService reads the active model from rag_config.yml and its API key from
    # the env var named by that model's api_key_env. We pass in the retriever
    # built above, so RagService skips loading the FAISS index a second time.
    rag = RagService(retriever=retriever)

    app.state.rag = rag
    app.state.ready = True

    elapsed = time.monotonic() - t0
    logger.info("chatbot ready in %.1fs", elapsed)

    yield

    # TODO: add cleanup here if RagService grows resources (open files, threads, etc.)
