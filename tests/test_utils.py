"""
 #! tests/test_utils.py
 summary:
    Testing utilities for mocking and simulation of RAG components.

 details:
    Provides mock classes for testing conversational RAG pipeline without requiring
    full FAISS indices or real document retrieval. Enables isolated testing and debugging
    of conversation history, truncation, and prompt construction.
"""

from typing import List
from langchain_core.documents import Document as LangchainDocument
from ceo_chatbot.rag.pipeline import RagService


class MockRetriever:
    """
    Mock document retriever for testing RAG pipeline.

    details:
        Returns a configurable list of documents for any query, simulating
        document retrieval without requiring FAISS or actual data.

    Args:
        documents (List[LangchainDocument]): List of documents to return for retrieval.

    Notes:
        Always returns the first k documents from the provided list.
    """

    def __init__(self, documents: List[LangchainDocument]):
        """
        Initialize with list of mock documents.
        """
        self.documents = documents

    def __call__(self, query: str, k: int) -> List[LangchainDocument]:
        """
        Retrieve mock documents.

        details:
            Matches the retriever interface for testing.

        Args:
            query (str): Ignored, but matches retriever interface.
            k (int): Number of documents to return.

        Returns:
            List[LangchainDocument]: Up to k mock documents.
        """
        return self.documents[:k]


class TestRagService(RagService):
    """
    Test version of RagService with mock document retrieval.

    details:
        Inherits from RagService but uses MockRetriever for testing.
        Allows full pipeline testing without FAISS indices. Supports
        artificial context limit overrides for truncation testing.

    Args:
        mock_docs (List[LangchainDocument]): Documents to return for retrieval.
        config_path (str): Path to config file for model/loading settings.
        max_context_tokens_override (int | None): Override max_context_tokens for testing.
        skip_interpretation (bool): If True, skips question interpretation for testing.

    Notes:
        This enables testing conversation history, token management, and prompt building
        in isolation from actual document databases. Useful for truncation validation.
    """

    def __init__(self, mock_docs: List[LangchainDocument], config_path: str = "conf/base/rag_config.yml", max_context_tokens_override: int | None = None, skip_interpretation: bool = False):
        """
        Initialize TestRagService with mock retriever.

        details:
            Creates MockRetriever and passes to parent RagService. Optionally overrides
            max_context_tokens for controlled truncation testing.

        Args:
            mock_docs (List[LangchainDocument]): Mock documents for retrieval.
            config_path (str): Config file path.
            max_context_tokens_override (int | None): Override max_context_tokens if provided.
            skip_interpretation (bool): If True, skips question interpretation for testing.
        """
        mock_retriever = MockRetriever(mock_docs)
        super().__init__(config_path=config_path, retriever=mock_retriever, skip_interpretation=skip_interpretation)

        if max_context_tokens_override is not None:
            self.max_context_tokens = max_context_tokens_override
