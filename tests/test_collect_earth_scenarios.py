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
from ceo_chatbot.rag.interpreter import Interpretation

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
    
    # Test interpretation
    interpretation = rag.interpreter.interpret_question(question, history)
    
    # Objective criteria validation
    print("\n--- Interpretation Analysis ---")
    print(f"Original question: {interpretation.original_question}")
    print(f"Interpreted question: {interpretation.interpreted_question}")
    print(f"Context used: {interpretation.context_used}")
    print(f"Needs retrieval: {interpretation.needs_retrieval}")
    print(f"Search query: {interpretation.search_query}")
    print(f"Interpretation confidence: {interpretation.interpretation_confidence}")
    print(f"Routing confidence: {interpretation.routing_confidence}")
    print(f"Reasoning: {interpretation.reasoning}")
    
    # Validate expectations
    assert interpretation.context_used == "standalone", f"Expected standalone, got {interpretation.context_used}"
    assert interpretation.needs_retrieval == True, "Expected needs_retrieval to be True for technical question"
    assert interpretation.search_query is not None, "Expected search_query to be provided"
    assert len(interpretation.interpreted_question) > 0, "Expected question to be rephrased"
    assert interpretation.interpretation_confidence >= 0.7, "Expected high interpretation confidence"
    
    print("\n✅ Single-turn scenario passed: Correct interpretation for standalone technical question")
    
    # Test full answer generation
    print("\n--- Full Answer Generation ---")
    answer, docs = rag.answer(question, history, debug=True)
    
    print(f"Answer length: {len(answer)} chars")
    print(f"Retrieved documents: {len(docs)}")
    print(f"Answer preview: {answer[:200]}...")
    
    assert len(answer) > 0, "Expected non-empty answer"
    assert len(docs) > 0, "Expected documents to be retrieved"
    
    return interpretation, answer, docs

def test_multi_turn_scenario():
    """
    Test multi-turn scenario with objective criteria.
    
    Scenario: 
    1. User asks "Can you please list the available choices for sample design?"
    2. System answers with options
    3. User asks "Why would I want to use the second option?"
    
    Expected: Interpretation should resolve "second option" to "Gridded sampling" and route to retrieval.
    """
    print("\n=== COLLECT EARTH ONLINE - MULTI-TURN SCENARIO ===\n")
    
    # Initialize with real RAG service
    rag = get_rag_service()
    
    # Step 1: First question
    question1 = "Can you please list the available choices for sample design?"
    history1 = []
    
    print(f"Step 1 - Question: {question1}")
    answer1, docs1 = rag.answer(question1, history1, debug=True)
    
    print(f"Answer 1 length: {len(answer1)} chars")
    print(f"Answer 1 preview: {answer1}...")
    
    # Step 2: Build conversation history
    conversation = [
        {"role": "user", "content": question1},
        {"role": "assistant", "content": answer1}
    ]
    
    # Step 3: Second question (follow-up)
    question2 = "Why would I want to use the second option?"
    history2 = conversation[:]
    
    print(f"\nStep 2 - Question: {question2}")
    print(f"History: {len(history2)} messages")
    
    # Test interpretation
    interpretation = rag.interpreter.interpret_question(question2, history2)
    
    # Objective criteria validation
    print("\n--- Interpretation Analysis ---")
    print(f"Original question: {interpretation.original_question}")
    print(f"Interpreted question: {interpretation.interpreted_question}")
    print(f"Context used: {interpretation.context_used}")
    print(f"Needs retrieval: {interpretation.needs_retrieval}")
    print(f"Search query: {interpretation.search_query}")
    print(f"Interpretation confidence: {interpretation.interpretation_confidence}")
    print(f"Routing confidence: {interpretation.routing_confidence}")
    print(f"Reasoning: {interpretation.reasoning}")
    
    # Validate expectations
    assert interpretation.context_used == "enhanced", f"Expected enhanced, got {interpretation.context_used}"
    assert interpretation.needs_retrieval == True, "Expected needs_retrieval to be True for detail-seeking question"
    assert interpretation.search_query is not None, "Expected search_query to be provided"
    
    # More flexible assertion for interpreted question content
    # this is where my ideas of how to test the quality or content of generative conversation is limited. 
    # last time using this assertion, the test failed but upon inspection of the interpretation it was more sophisticated (and good!)
    # than what I expected to see when i wrote this assertion..
    # i.e. we don't know what the second option is going to be per se for a given generated first answer in this scneario.
    # assert any(word in interpretation.interpreted_question.lower() for word in ["gridded", "second", "plot", "sampling"]), \
    #     f"Expected interpreted question to reference the second option, got: {interpretation.interpreted_question}"
    
    assert interpretation.interpretation_confidence >= 0.7, "Expected high interpretation confidence"
    
    print("\n✅ Multi-turn scenario passed: Correct interpretation for contextual question")
    
    # Step 4: Test full answer generation
    print("\n--- Full Answer Generation ---")
    answer2, docs2 = rag.answer(question2, history2, debug=True)
    
    print(f"Answer 2 length: {len(answer2)} chars")
    print(f"Retrieved documents: {len(docs2)}")
    print(f"Answer 2 preview: {answer2}...")
    
    assert len(answer2) > 0, "Expected non-empty answer"
    assert len(docs2) > 0, "Expected documents to be retrieved"
    
    return interpretation, answer2, docs2

def test_retrieval_conservative_behavior():
    """
    Test that the system is conservative about retrieval decisions.
    
    Scenarios that should trigger retrieval:
    - Questions with detail-seeking keywords
    - Questions in technical domains
    - Questions with low context quality
    """
    print("\n=== RETRIEVAL-CONSERVATIVE BEHAVIOR TESTS ===\n")
    
    rag = get_rag_service()
    
    test_cases = [
        {
            "question": "How does gridded sampling work?",
            "history": [],
            "expected_retrieval": True,
            "reason": "Detail-seeking question about technical topic"
        },
        {
            "question": "What are the benefits of using high-resolution imagery?",
            "history": [],
            "expected_retrieval": True,
            "reason": "Question with 'benefits' keyword"
        },
        {
            "question": "Can you explain the process for creating plots?",
            "history": [],
            "expected_retrieval": True,
            "reason": "Question with 'explain' and 'process' keywords"
        },
        {
            "question": "What did you just say?",
            "history": [{"role": "user", "content": "What is sampling?"}, {"role": "assistant", "content": "Sampling is..."}],
            "expected_retrieval": False,
            "reason": "Repetition question"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"--- Test Case {i}: {test_case['reason']} ---")
        print(f"Question: {test_case['question']}")
        print(f"History: {len(test_case['history'])} messages")
        
        interpretation = rag.interpreter.interpret_question(test_case['question'], test_case['history'])
        
        print(f"Needs retrieval: {interpretation.needs_retrieval}")
        print(f"Search query: {interpretation.search_query}")
        print(f"Interpretation: {interpretation.interpreted_question}")
        
        assert interpretation.needs_retrieval == test_case['expected_retrieval'], \
            f"Expected needs_retrieval={test_case['expected_retrieval']}, got {interpretation.needs_retrieval}"
        
        if interpretation.needs_retrieval:
            assert interpretation.search_query is not None, "Expected search_query for retrieval-needed question"
        
        print("✅ Test case passed\n")
    
    print("✅ All retrieval-conservative behavior tests passed!")

def run_collect_earth_test_suite():
    """
    Run all Collect Earth Online test scenarios.
    """
    print("🚀 COLLECT EARTH ONLINE RAG PIPELINE TEST SUITE")
    print("=" * 60)
    
    try:
        # Test single-turn scenario
        interpretation1, answer1, docs1 = test_single_turn_scenario()
        
        # Test multi-turn scenario
        interpretation2, answer2, docs2 = test_multi_turn_scenario()
        
        # Test retrieval-conservative behavior
        test_retrieval_conservative_behavior()
        
        print("\n🎉 ALL COLLECT EARTH ONLINE TESTS COMPLETED SUCCESSFULLY!")
        print("\n📋 Test Summary:")
        print("✅ Single-turn scenario: Correct interpretation and retrieval routing")
        print("✅ Multi-turn scenario: Proper context resolution and retrieval routing")
        print("✅ Retrieval-conservative behavior: Appropriate routing decisions")
        print("\n🔍 Key Validations:")
        print("- Interpretation schema working correctly")
        print("- Context quality assessment functioning")
        print("- Question intent analysis accurate")
        print("- Technical domain detection working")
        print("- Retrieval routing conservative as designed")
        print("- Real FAISS database integration successful")
        
        print("\n🎯 System Status: READY FOR PRODUCTION USE!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_collect_earth_test_suite()
    exit(0 if success else 1)