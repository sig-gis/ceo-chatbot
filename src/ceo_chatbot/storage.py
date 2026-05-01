import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSStorage:
    """Thin wrapper around google-cloud-storage for this project's access patterns."""

    def __init__(self, bucket_name: str) -> None:
        self._client = storage.Client()
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

        for local_path in (f for f in local_dir.rglob("*") if f.is_file()):
            rel = local_path.relative_to(local_dir)
            remote_path = f"{remote_prefix}/{rel}"
            blob = self._bucket.blob(remote_path)

            if not blob.exists():
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
        blobs = list(self._client.list_blobs(self._bucket, prefix=remote_prefix))
        for blob in blobs:
            rel = blob.name[len(remote_prefix):].lstrip("/")
            self.download(blob.name, local_dir / rel)
        return len(blobs)

    def upload(self, local: Path, remote: str) -> None:
        """Upload a single file unconditionally."""
        self._bucket.blob(remote).upload_from_filename(local)

    def blob_updated(self, remote: str) -> datetime | None:
        """Return the GCS last-modified timestamp for a blob, or None if it does not exist.

        blob() returns a stub with no metadata; reload() fetches .updated from GCS.
        """
        blob = self._bucket.blob(remote)
        if not blob.exists():
            return None
        blob.reload()
        return blob.updated

    def exists(self, remote: str) -> bool:
        """Return True if the blob exists in the bucket."""
        return self._bucket.blob(remote).exists()

    def list(self, prefix: str = "") -> list[str]:
        """Return all blob names under prefix (or all blobs if prefix is empty)."""
        return [b.name for b in self._client.list_blobs(self._bucket, prefix=prefix)]
