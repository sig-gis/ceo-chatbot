"""Tests for GCSStorage methods beyond sync_up (covered in test_storage.py)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from ceo_chatbot_core.storage import GCSStorage


def _make_store(mock_client_cls, bucket_name="test-bucket"):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return GCSStorage(bucket_name), mock_client, mock_bucket


@patch("ceo_chatbot_core.storage.storage.Client")
def test_upload_calls_gcs(mock_client_cls, tmp_path):
    """Verifies upload() targets the right blob name and calls upload_from_filename."""
    store, _, mock_bucket = _make_store(mock_client_cls)
    f = tmp_path / "file.txt"
    f.write_text("data")
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    store.upload(f, "prefix/file.txt")

    mock_bucket.blob.assert_called_once_with("prefix/file.txt")
    mock_blob.upload_from_filename.assert_called_once_with(f)


@patch("ceo_chatbot_core.storage.storage.Client")
def test_download_creates_parent_dirs_and_calls_gcs(mock_client_cls, tmp_path):
    """Confirms download() creates missing parent directories before writing the file."""
    store, _, mock_bucket = _make_store(mock_client_cls)
    dest = tmp_path / "deep" / "dir" / "file.txt"
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    store.download("prefix/file.txt", dest)

    assert dest.parent.exists()
    mock_blob.download_to_filename.assert_called_once_with(dest)


@patch("ceo_chatbot_core.storage.storage.Client")
def test_download_prefix_returns_count(mock_client_cls, tmp_path):
    """download_prefix() returns the number of blobs downloaded."""
    store, mock_client, mock_bucket = _make_store(mock_client_cls)
    blob_a, blob_b = MagicMock(), MagicMock()
    blob_a.name = "docs/a.txt"
    blob_b.name = "docs/b.txt"
    mock_client.list_blobs.return_value = [blob_a, blob_b]
    mock_bucket.blob.return_value = MagicMock()

    count = store.download_prefix("docs", tmp_path)

    assert count == 2
    mock_client.list_blobs.assert_called_once_with(mock_bucket, prefix="docs")


@patch("ceo_chatbot_core.storage.storage.Client")
def test_download_prefix_strips_prefix_from_local_path(mock_client_cls, tmp_path):
    """The remote prefix is stripped before building the local path.

    docs/sub/file.txt downloaded to local_dir lands at local_dir/sub/file.txt,
    not local_dir/docs/sub/file.txt.
    """
    store, mock_client, mock_bucket = _make_store(mock_client_cls)
    blob = MagicMock()
    blob.name = "docs/sub/file.txt"
    mock_client.list_blobs.return_value = [blob]
    mock_blob_instance = MagicMock()
    mock_bucket.blob.return_value = mock_blob_instance

    store.download_prefix("docs", tmp_path)

    expected_local = tmp_path / "sub" / "file.txt"
    mock_blob_instance.download_to_filename.assert_called_once_with(expected_local)


@patch("ceo_chatbot_core.storage.storage.Client")
def test_exists_delegates_to_blob(mock_client_cls):
    """exists() returns True/False directly from blob.exists() with no extra logic."""
    store, _, mock_bucket = _make_store(mock_client_cls)
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_blob.exists.return_value = True
    assert store.exists("file.txt") is True

    mock_blob.exists.return_value = False
    assert store.exists("other.txt") is False


@patch("ceo_chatbot_core.storage.storage.Client")
def test_list_returns_blob_names(mock_client_cls):
    """list() returns a flat list of blob name strings under the given prefix."""
    store, mock_client, mock_bucket = _make_store(mock_client_cls)
    b1, b2 = MagicMock(), MagicMock()
    b1.name = "docs/a.txt"
    b2.name = "docs/b.txt"
    mock_client.list_blobs.return_value = [b1, b2]

    result = store.list("docs")

    assert result == ["docs/a.txt", "docs/b.txt"]
    mock_client.list_blobs.assert_called_once_with(mock_bucket, prefix="docs")
