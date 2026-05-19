"""Tests for URL calculation logic in load_rst_docs.

UnstructuredRSTLoader is mocked so tests run without the unstructured package
performing actual RST parsing and without real RST file content.
"""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document as LangchainDocument

from ceo_build_index.loaders import load_rst_docs

BASE_URL = "https://collect-earth-online-doc.readthedocs.io/en/latest/"


def _mock_loader_cls(docs):
    """Return a mock UnstructuredRSTLoader class that yields `docs` on .load()."""
    cls = MagicMock()
    cls.return_value.load.return_value = docs
    return cls


def _doc():
    return LangchainDocument(page_content="text", metadata={})


def test_empty_directory_returns_empty_list(tmp_path):
    """A directory with no .rst files returns an empty list without error."""
    assert load_rst_docs(str(tmp_path)) == []


def test_url_uses_html_extension(tmp_path):
    """The .rst extension is converted to .html in every generated URL."""
    (tmp_path / "guide.rst").write_text("")
    with patch("ceo_build_index.loaders.rst.UnstructuredRSTLoader", _mock_loader_cls([_doc()])):
        results = load_rst_docs(str(tmp_path))
    assert results[0].metadata["url"].endswith(".html")
    assert ".rst" not in results[0].metadata["url"]


def test_url_strips_source_from_path(tmp_path):
    """When source/ appears in the file path, it is stripped and does not appear in the URL."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "index.rst").write_text("")
    with patch("ceo_build_index.loaders.rst.UnstructuredRSTLoader", _mock_loader_cls([_doc()])):
        results = load_rst_docs(str(tmp_path))
    suffix = results[0].metadata["url"][len(BASE_URL):]
    assert not suffix.startswith("source")
    assert suffix == "index.html"


def test_url_uses_relative_path_when_no_source_dir(tmp_path):
    """Without a source/ directory, the URL is built from the path relative to docs_dir."""
    sub = tmp_path / "institution"
    sub.mkdir()
    (sub / "imagery.rst").write_text("")
    with patch("ceo_build_index.loaders.rst.UnstructuredRSTLoader", _mock_loader_cls([_doc()])):
        results = load_rst_docs(str(tmp_path))
    assert results[0].metadata["url"] == f"{BASE_URL}institution/imagery.html"


def test_url_set_on_every_element_from_one_file(tmp_path):
    """All elements produced by a single RST file share the same URL in their metadata."""
    (tmp_path / "page.rst").write_text("")
    docs = [_doc(), _doc(), _doc()]
    with patch("ceo_build_index.loaders.rst.UnstructuredRSTLoader", _mock_loader_cls(docs)):
        results = load_rst_docs(str(tmp_path))
    assert len(results) == 3
    urls = {d.metadata["url"] for d in results}
    assert len(urls) == 1
