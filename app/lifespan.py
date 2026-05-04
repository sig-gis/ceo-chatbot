import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ceo_chatbot.config import load_rag_config
from ceo_chatbot.rag.pipeline import RagService
from ceo_chatbot.rag.retriever import get_retriever, load_faiss_index
from ceo_chatbot.storage import GCSStorage
from app.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download the FAISS index from GCS and initialise RagService on startup."""
    t0 = time.monotonic()
    settings = get_settings()

    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    # Download index.faiss and index.pkl from GCS into the local directory.
    # Cloud Run containers start fresh, so the files are never present on startup.
    # TODO: add smart-load (skip download if local copy is newer than GCS blob)
    #       using GCSStorage.blob_updated() once we want to support local dev speed.
    settings.index_local_dir.mkdir(parents=True, exist_ok=True)
    config = load_rag_config()
    
    # Prefix comes from rag_config.yml (vectorstore_gcs) — same value the pipeline
    # used when uploading, so the download path always matches the upload path.
    index_prefix = str(config.vectorstore_gcs)
    gcs = GCSStorage(settings.gcs_bucket_index, project=settings.google_cloud_project)
    for fname in ("index.faiss", "index.pkl"):
        gcs.download(
            f"{index_prefix}/{fname}",
            settings.index_local_dir / fname,
        )
    logger.info("index downloaded to %s", settings.index_local_dir)

    # Build the retriever from the downloaded index. We inject it into RagService
    # so RagService loads from settings.index_local_dir rather than the
    # vectorstore_path in rag_config.yml — without needing to change pipeline.py.
    faiss_index = load_faiss_index(settings.index_local_dir, config.embedding_model_name)
    retriever = get_retriever(faiss_index)

    # RagService reads LLM settings from rag_config.yml and picks up GEMINI_API_KEY
    # from the environment via GeminiInferenceSettings. The injected retriever
    # bypasses its internal FAISS load.
    rag = RagService(retriever=retriever)

    app.state.rag = rag
    app.state.ready = True

    elapsed = time.monotonic() - t0
    logger.info("chatbot ready in %.1fs", elapsed)

    yield

    # TODO: add cleanup here if RagService grows resources (open files, threads, etc.)
