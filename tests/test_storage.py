# Test module for GCSStorage with unit tests for sync_up functionality
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from ceo_chatbot.storage import GCSStorage


def _make_store(mock_client_cls, bucket_name="test-bucket"):
    """Helper function to create a mock GCSStorage instance with mocked GCS client and bucket."""
    # Create a mock client and set it as the return value of the mocked Client class
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    # Create a mock bucket and set it as the return value of client.bucket()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    return GCSStorage(bucket_name), mock_bucket


def _make_blob(*, exists, size, remote_updated_offset_hours):
    """Helper function to create a mock blob with specified properties."""
    # Create a mock blob object
    blob = MagicMock()
    # Set the exists property to return the specified value
    blob.exists.return_value = exists
    # Set the size attribute
    blob.size = size
    # Set the updated timestamp with offset from current UTC time
    blob.updated = datetime.now(tz=timezone.utc) + timedelta(hours=remote_updated_offset_hours)
    return blob


@patch("ceo_chatbot.storage.storage.Client")
def test_sync_up_uploads_when_blob_missing(mock_client_cls, tmp_path):
    """Test that sync_up uploads a file when the remote blob does not exist."""
    # Set up mock storage with a mocked client
    store, mock_bucket = _make_store(mock_client_cls)
    # Create a test file
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    # Create a mock blob that doesn't exist
    blob = _make_blob(exists=False, size=0, remote_updated_offset_hours=0)
    mock_bucket.blob.return_value = blob

    # Execute sync_up
    result = store.sync_up(tmp_path, "prefix")

    # Verify that upload was called and the result shows 1 uploaded file
    blob.upload_from_filename.assert_called_once_with(f)
    assert result == {"uploaded": 1, "skipped": 0, "total": 1}


@patch("ceo_chatbot.storage.storage.Client")
def test_sync_up_skips_when_size_equal_and_local_newer(mock_client_cls, tmp_path):
    """Test that sync_up skips a file when sizes match and local file is newer."""
    # Set up mock storage
    store, mock_bucket = _make_store(mock_client_cls)
    # Create a test file
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    # Create a mock blob with matching size but older timestamp (remote is 1 hour old)
    blob = _make_blob(exists=True, size=f.stat().st_size, remote_updated_offset_hours=-1)
    mock_bucket.blob.return_value = blob

    # Execute sync_up
    result = store.sync_up(tmp_path, "prefix")

    # Verify that upload was NOT called and the result shows 1 skipped file
    blob.upload_from_filename.assert_not_called()
    assert result == {"uploaded": 0, "skipped": 1, "total": 1}


@patch("ceo_chatbot.storage.storage.Client")
def test_sync_up_uploads_when_size_differs(mock_client_cls, tmp_path):
    """Test that sync_up uploads a file when sizes differ between local and remote."""
    # Set up mock storage
    store, mock_bucket = _make_store(mock_client_cls)
    # Create a test file
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    # Create a mock blob with different size and older timestamp
    blob = _make_blob(exists=True, size=f.stat().st_size + 1, remote_updated_offset_hours=-1)
    mock_bucket.blob.return_value = blob

    # Execute sync_up
    result = store.sync_up(tmp_path, "prefix")

    # Verify that upload was called and the result shows 1 uploaded file
    blob.upload_from_filename.assert_called_once_with(f)
    assert result == {"uploaded": 1, "skipped": 0, "total": 1}


@patch("ceo_chatbot.storage.storage.Client")
def test_sync_up_skips_when_local_older(mock_client_cls, tmp_path):
    """Test that sync_up skips a file when the local version is older than remote."""
    # Set up mock storage
    store, mock_bucket = _make_store(mock_client_cls)
    # Create a test file
    f = tmp_path / "doc.txt"
    f.write_text("hello")
    # Create a mock blob with newer timestamp (remote is 1 hour in the future)
    blob = _make_blob(exists=True, size=f.stat().st_size, remote_updated_offset_hours=1)
    mock_bucket.blob.return_value = blob

    # Execute sync_up
    result = store.sync_up(tmp_path, "prefix")

    # Verify that upload was NOT called and the result shows 1 skipped file
    blob.upload_from_filename.assert_not_called()
    assert result == {"uploaded": 0, "skipped": 1, "total": 1}


@patch("ceo_chatbot.storage.storage.Client")
def test_sync_up_returns_correct_counts(mock_client_cls, tmp_path):
    """Test that sync_up returns correct counts when syncing multiple files with different conditions."""
    # Set up mock storage
    store, mock_bucket = _make_store(mock_client_cls)

    # Create three test files
    f1 = tmp_path / "a.txt"
    f1.write_text("aaa")  # missing → upload

    f2 = tmp_path / "b.txt"
    f2.write_text("bbb")  # local newer, size differs → upload

    f3 = tmp_path / "c.txt"
    f3.write_text("ccc")  # local older → skip

    # Set up mock blobs for each file with different conditions
    blobs = {
        "prefix/a.txt": _make_blob(exists=False, size=0, remote_updated_offset_hours=0),
        "prefix/b.txt": _make_blob(exists=True, size=f2.stat().st_size + 10, remote_updated_offset_hours=-1),
        "prefix/c.txt": _make_blob(exists=True, size=f3.stat().st_size, remote_updated_offset_hours=1),
    }
    # Configure mock to return appropriate blob for each file name
    mock_bucket.blob.side_effect = lambda name: blobs[name]

    # Execute sync_up
    result = store.sync_up(tmp_path, "prefix")

    # Verify that the result shows correct counts: 2 uploaded, 1 skipped, 3 total
    assert result == {"uploaded": 2, "skipped": 1, "total": 3}
