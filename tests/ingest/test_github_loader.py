import subprocess

import pytest
from unittest.mock import patch, MagicMock

from ceo_ingest_docs.config import IngestionConfig
from ceo_ingest_docs.github_loader import GitHubLoader


def _loader(repo_url="https://github.com/org/repo.git", ref="main", path=""):
    return GitHubLoader(IngestionConfig(repo_url=repo_url, ref=ref, path=path))


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_calls_git_with_correct_args(mock_run, tmp_path):
    """The subprocess call uses git clone --depth 1 --branch <ref> with the configured URL and target path."""
    _loader().clone(tmp_path)

    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "git"
    assert "--depth" in cmd and "1" in cmd
    assert "https://github.com/org/repo.git" in cmd
    assert str(tmp_path / "repo.git") in cmd


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_extracts_repo_name_from_url(mock_run, tmp_path):
    """The repository name is parsed from the last segment of the URL path and used as the clone directory."""
    result = _loader(repo_url="https://github.com/org/my-repo.git").clone(tmp_path)
    assert result == tmp_path / "my-repo.git"


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_returns_full_repo_when_no_sub_path(mock_run, tmp_path):
    """When config.path is empty, the clone root directory is returned."""
    result = _loader().clone(tmp_path)
    assert result == tmp_path / "repo.git"


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_returns_sub_path_when_configured(mock_run, tmp_path):
    """When config.path is set and the directory exists, that sub-path is returned."""
    sub = tmp_path / "repo.git" / "docs" / "source"
    sub.mkdir(parents=True)

    result = _loader(path="docs/source").clone(tmp_path)

    assert result == sub


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_raises_runtime_error_on_git_failure(mock_run, tmp_path):
    """A non-zero git exit code raises RuntimeError('Failed to clone…')."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "git", stderr="auth failed")

    with pytest.raises(RuntimeError, match="Failed to clone"):
        _loader().clone(tmp_path)


@patch("ceo_ingest_docs.github_loader.subprocess.run")
def test_clone_raises_when_sub_path_missing(mock_run, tmp_path):
    """If the configured sub-path is not found inside the cloned repo, raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        _loader(path="nonexistent/path").clone(tmp_path)
