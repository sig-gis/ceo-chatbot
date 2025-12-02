import subprocess
from pathlib import Path
from ..config import DocumentExtractionConfig


class GCSUploader:
    """
    Handles uploading documents from a local directory to Google Cloud Storage.
    """

    def __init__(self, config: DocumentExtractionConfig):
        self.config = config

    def upload(self, local_path: Path) -> None:
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
        gcs_path = f"gs://{self.config.gcs_bucket_name}"
        if self.config.gcs_prefix:
            gcs_path += f"/{self.config.gcs_prefix}"

        # Use gsutil to recursively copy the files
        # -r for recursive, -m for parallel upload for better performance
        cmd = ["gsutil", "-m", "cp", "-r", str(local_path), gcs_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to upload to GCS: {e.stderr}") from e
