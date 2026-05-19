from unittest.mock import MagicMock

from langchain_core.documents import Document as LangchainDocument

from ceo_rag_chatbot.rag.retriever import get_retriever


def test_retriever_calls_similarity_search_with_correct_args():
    """The callable forwards query and k to index.similarity_search and returns its result."""
    doc = LangchainDocument(page_content="content", metadata={"url": "https://example.com"})
    mock_index = MagicMock()
    mock_index.similarity_search.return_value = [doc]

    retrieve = get_retriever(mock_index)
    results = retrieve("what is CEO?", k=3)

    mock_index.similarity_search.assert_called_once_with(query="what is CEO?", k=3)
    assert results == [doc]


def test_retriever_default_k_is_five():
    """Calling the retriever without an explicit k uses k=5."""
    mock_index = MagicMock()
    mock_index.similarity_search.return_value = []

    retrieve = get_retriever(mock_index)
    retrieve("query")

    _, kwargs = mock_index.similarity_search.call_args
    assert kwargs["k"] == 5


def test_retriever_returns_empty_list_when_no_matches():
    """An empty result from similarity_search propagates cleanly as []."""
    mock_index = MagicMock()
    mock_index.similarity_search.return_value = []

    retrieve = get_retriever(mock_index)
    assert retrieve("obscure question", k=5) == []
