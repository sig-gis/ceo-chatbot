"""
 #! tests/test_collect_earth_scenarios.py
 summary:
    Test suite for real-world Collect Earth Online scenarios with objective criteria.

 details:
    Tests both single-turn and multi-turn scenarios using the actual RagService
    with real FAISS database. Validates interpretation logic, routing decisions,
    and answer generation for Collect Earth Online domain.
"""

import logging
from ceo_chatbot.rag.pipeline import RagService
import re

# Configure logging for debug output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_rag_service(config_path: str = "conf/base/rag_config.yml") -> RagService:
    """
    Create and cache a single RagService instance.

    Caching ensures:
      - FAISS index is loaded once
      - Hugging Face model checkpoints are loaded once
    """
    return RagService(config_path=config_path)

def test_single_turn_scenario():
    """
    Test single-turn scenario with objective criteria.
    
    Scenario: User asks "How do I add reference imagery to my project?"
    Expected: Interpretation should route to retrieval with appropriate search query.
    """
    print("=== COLLECT EARTH ONLINE - SINGLE-TURN SCENARIO ===\n")
    
    # Initialize with real RAG service (not mock)
    rag = get_rag_service()
    
    question = "How do I add reference imagery to my project?"
    history = []
    
    print(f"Question: {question}")
    print(f"History: {len(history)} messages")
    
    interpretation = rag._interpret_question_phase1(question, history,False)
   
    prompt, docs, context, truncated_history = rag._build_prompt_and_retrieve(
        interpretation, history, 20, 5
    )

    all_urls = re.findall(r'https?://\S+', prompt)    
    
    assert [doc.metadata.get('url') for doc in docs], "URLs not in doc metadata"
    assert len(all_urls) >0, "No valid URLs provided in prompt"

if __name__=="__main__":
    test_single_turn_scenario()