from ceo_chatbot.ingest.gcs_uploader import GCSHandler
from ceo_chatbot.config import load_rag_config

gcshandle = GCSHandler(load_rag_config())
gcshandle.init_db()