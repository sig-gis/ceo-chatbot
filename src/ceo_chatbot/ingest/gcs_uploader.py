import subprocess
from pathlib import Path
from ..config import RAGConfig, AppSettings
class GCSHandler:
    """
    Handles file syncing between local machine and Google Cloud Storage
    """

    def __init__(self, 
                 rag_config: RAGConfig):
        
        self.app_settings = AppSettings()
        self.rag_config = rag_config

    def upload_docs(self, local_path: Path) -> str | Path:
        """
        Upload all files from the local path to the GCS bucket.

        Args:
            local_path: Local directory or file to upload.

        Raises:
            subprocess.CalledProcessError: If gsutil cp fails.
            FileNotFoundError: If local_path does not exist.
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local path '{local_path}' does not exist.")

        # Determine the GCS destination
        gcs_bucket = f"gs://{self.app_settings.docs_bucket_name}"
        prefix = self.app_settings.folder_prefix
        if prefix:
            gcs_path = f"{gcs_bucket}/{prefix}"

        # sync docs/source folder between github repo and GCS bucket's copy
        cmd = ["gsutil", "rsync", "-r", str(local_path), gcs_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to upload to GCS: {e.stderr}") from e
        print(f"ceo-docs data synced from {local_path} to {gcs_path}")

        return gcs_path
    
    def upload_db(self, local_path: str | Path = "data/vectorstores/ceo_docs_faiss") -> str | Path:
        """
        Upload all FAISS vector database files from the local path to the GCS bucket.

        Args:
            local_path: Local directory or file to upload.

        Raises:
            subprocess.CalledProcessError: If gsutil cp fails.
            FileNotFoundError: If local_path does not exist.
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local path '{local_path}' does not exist.")

        # Determine the GCS destination
        gcs_bucket = f"gs://{self.app_settings.db_bucket_name}"
        if self.rag_config.vectorstore_gcs:
            gcs_path = f"{gcs_bucket}/{self.rag_config.vectorstore_gcs}"

        # sync docs/source folder between github repo and GCS bucket's copy
        cmd = ["gsutil", "rsync", "-r", str(local_path), gcs_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to upload to GCS: {e.stderr}") from e

        print(f"FAISS DB files synced from {local_path} to {gcs_path}")
        return gcs_path
        
    def init_db(self, dest_path: str | Path = "data/vectorstores/ceo_docs_faiss") -> str | Path:
        """ sync down the FAISS DB from GCS """
        if isinstance(dest_path,str):
            dest_path = Path(dest_path)
        
        dest_path.mkdir(exist_ok=True, parents=True)
        
        # Determine the GCS destination
        gcs_bucket = f"gs://{self.app_settings.db_bucket_name}"
        if self.rag_config.vectorstore_gcs:
            gcs_path = f"{gcs_bucket}/{self.rag_config.vectorstore_gcs}"

        # sync docs/source folder between github repo and GCS bucket's copy
        cmd = ["gsutil", "rsync", "-r", gcs_path, str(dest_path)]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to upload to GCS: {e.stderr}") from e

        print(f"FAISS DB files synced from {gcs_path} to {dest_path}")
        return dest_path