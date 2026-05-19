import pytest
from pydantic import ValidationError

from ceo_rag_chatbot.rag.interpreter import Interpretation


def _base(**kwargs):
    defaults = dict(
        original_question="What is CEO?",
        interpreted_question="What is CEO?",
        interpreted_successfully=True,
        interpretation_confidence=0.9,
        routing_confidence=0.8,
        needs_retrieval=True,
        search_query="CEO definition",
        context_used="standalone",
        reasoning="Question asks about CEO directly.",
    )
    defaults.update(kwargs)
    return defaults


def test_valid_retrieval_interpretation():
    """A complete, valid interpretation with needs_retrieval=True is accepted."""
    interp = Interpretation(**_base())
    assert interp.needs_retrieval is True
    assert interp.search_query == "CEO definition"


def test_valid_non_retrieval_interpretation():
    """A valid interpretation with needs_retrieval=False and search_query=None is accepted."""
    interp = Interpretation(**_base(needs_retrieval=False, search_query=None))
    assert interp.search_query is None


def test_search_query_required_when_retrieval_needed():
    """search_query=None with needs_retrieval=True raises ValidationError."""
    with pytest.raises(ValidationError, match="search_query required"):
        Interpretation(**_base(needs_retrieval=True, search_query=None))


def test_search_query_must_be_none_when_no_retrieval():
    """A non-null search_query with needs_retrieval=False raises ValidationError."""
    with pytest.raises(ValidationError, match="search_query should be None"):
        Interpretation(**_base(needs_retrieval=False, search_query="some query"))


def test_context_used_must_be_standalone_or_enhanced():
    """Only 'standalone' and 'enhanced' are valid values for context_used."""
    Interpretation(**_base(context_used="standalone"))
    Interpretation(**_base(context_used="enhanced"))
    with pytest.raises(ValidationError):
        Interpretation(**_base(context_used="unknown"))


def test_confidence_scores_bounded():
    """Confidence scores above 1.0 or below 0.0 raise ValidationError."""
    with pytest.raises(ValidationError):
        Interpretation(**_base(interpretation_confidence=1.1))
    with pytest.raises(ValidationError):
        Interpretation(**_base(routing_confidence=-0.1))
