import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSStorage:
    """Thin wrapper around google-cloud-storage for this project's access patterns."""

    def __init__(self, bucket_name: str, project: str | None = None) -> None:
        self._client = storage.Client(project=project)
        self._bucket = self._client.bucket(bucket_name)

    def sync_up(self, local_dir: Path, remote_prefix: str) -> dict:
        """Mirror a local directory tree to gs://<bucket>/<remote_prefix>/.

        Per-file decision:
        - Remote blob missing → upload.
        - Local mtime newer than blob.updated AND sizes differ → upload.
        - Local mtime newer than blob.updated AND sizes equal → skip (touched, unchanged).
        - Local not newer → skip.

        Returns {"uploaded": int, "skipped": int, "total": int}.
        """
        uploaded = skipped = 0

        # For every file in local_dir - which is the tempdir where the docs were downloaded to 
        for local_path in (f for f in local_dir.rglob("*") if f.is_file()):
            rel = local_path.relative_to(local_dir) # Get the filename relative to tmpdir path
            remote_path = f"{remote_prefix}/{rel}" # Where the file will sit in GCS
            blob = self._bucket.blob(remote_path) # Create a stub blob

            if not blob.exists(): # Upload unconditonally if blob does not exist
                self.upload(local_path, remote_path)
                logger.info("uploaded %s → %s", local_path, remote_path)
                uploaded += 1
                continue

            # blob() returns a stub; reload() fetches .size and .updated from GCS
            blob.reload()

            stat = local_path.stat()
            # st_mtime is a naive Unix timestamp; make it tz-aware before comparing
            # to blob.updated, which is tz-aware UTC
            local_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            if local_mtime <= blob.updated:
                logger.debug("skipped %s (remote is newer or same age)", local_path)
                skipped += 1
                continue

            if stat.st_size == blob.size:
                logger.debug("skipped %s (same size, just touched)", local_path)
                skipped += 1
                continue
            
            # Upload document if remote blob is older than current version of it
            self.upload(local_path, remote_path)
            logger.info("uploaded %s → %s (size changed)", local_path, remote_path)
            uploaded += 1

        total = uploaded + skipped
        return {"uploaded": uploaded, "skipped": skipped, "total": total}

    def download(self, remote: str, local: Path) -> None:
        """Download a single blob to local."""
        local.parent.mkdir(parents=True, exist_ok=True)
        self._bucket.blob(remote).download_to_filename(local)

    def download_prefix(self, remote_prefix: str, local_dir: Path) -> int:
        """Download all blobs under remote_prefix into local_dir. Returns count."""
        # remote_prefix is something like `collect-earth-online-doc/docs/source`, local_dir is `data/ceo-docs`
        # The source files are located at DOCS_BUCKET/PREFIX
        blobs = list(self._client.list_blobs(self._bucket, prefix=remote_prefix))
        for blob in blobs:
            rel = blob.name[len(remote_prefix):].lstrip("/")
            self.download(blob.name, local_dir / rel)
        return len(blobs)

    def upload(self, local: Path, remote: str) -> None:
        """Upload a single file unconditionally."""
        self._bucket.blob(remote).upload_from_filename(local)

    def exists(self, remote: str) -> bool:
        """Return True if the blob exists in the bucket."""
        return self._bucket.blob(remote).exists()

    def list(self, prefix: str = "") -> list[str]:
        """Return all blob names under prefix (or all blobs if prefix is empty)."""
        return [b.name for b in self._client.list_blobs(self._bucket, prefix=prefix)]
