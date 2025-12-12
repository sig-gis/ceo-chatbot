"""
 #! tests/simulate_conversation.py
 summary:
    Simulation script for testing conversational RAG pipeline logic.

 details:
    Runs test scenarios with mock documents to validate conversation history,
    token truncation, prompt construction, and answer generation.
    Uses debug logging to inspect internal pipeline steps.
"""

import logging
from langchain_core.documents import Document as LangchainDocument
from test_utils import TestRagService

# Configure logging for debug output
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create mock documents for testing
mock_docs = [
    LangchainDocument(
        page_content="Python is a high-level programming language known for its simplicity and readability.",
        metadata={"source": "doc1"}
    ),
    LangchainDocument(
        page_content="RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation.",
        metadata={"source": "doc2"}
    ),
    LangchainDocument(
        page_content="Machine learning models often use vector databases like FAISS for efficient similarity search.",
        metadata={"source": "doc3"}
    ),
    LangchainDocument(
        page_content="Context length limits in LLMs restrict the number of tokens that can be processed in a single inference.",
        metadata={"source": "doc4"}
    ),
]

def run_simulation():
    """
    Run conversation simulation scenarios.
    """
    print("=== RAG Conversation Simulation ===\n")
    print(f"Mock documents available: {len(mock_docs)}")
    print(f"Document contents: {[doc.page_content[:50] + '...' for doc in mock_docs]}\n")

    # Initialize test RAG service
    rag = TestRagService(mock_docs)

    # Scenario 1: Single-turn conversation (baseline)
    print("--- Scenario 1: Single-turn conversation ---")
    conversation = []

    question = "What is Python?"
    history = []

    print(f"Input History messages: {len(history)}")
    answer, docs = rag.answer(question, history, debug=True)
    print(f"Q: {question}")
    print(f"A: {answer}\n")

    conversation.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])

    # Scenario 2: Short conversation (2 turns)
    print("--- Scenario 2: Short conversation ---")

    question2 = "How does RAG work?"
    history2 = conversation[:]  # Full history up to now

    print(f"Input History messages: {len(history2)}")
    answer2, docs2 = rag.answer(question2, history2, debug=True)
    print(f"Q: {question2}")
    print(f"A: {answer2}\n")

    conversation.extend([
        {"role": "user", "content": question2},
        {"role": "assistant", "content": answer2}
    ])

    # Scenario 3: Conversation with multiple turns (real context)
    print("--- Scenario 3: Long conversation (within real context limits) ---")
    # Add simulated history with realistic messages
    extended_history = conversation[:] + [
        {"role": "user", "content": "Tell me more about machine learning models and how they handle large datasets with vector embeddings."},
        {"role": "assistant", "content": "Machine learning models process data through algorithms. Vector embeddings are numerical representations that capture semantic meaning. Large datasets require scalable storage solutions like distributed databases."},
        {"role": "user", "content": "How do modern language models handle multi-turn conversations?"},
        {"role": "assistant", "content": "Modern LLMs use attention mechanisms and context windows. They maintain coherence through conversation history inclusion and prompt engineering."},
    ]

    question3 = "How does tokenization work?"
    history3 = extended_history[:]

    print(f"Input History messages: {len(history3)}")
    answer3, docs3 = rag.answer(question3, history3, debug=True)
    print(f"Q: {question3}")
    print(f"A: {answer3}\n")

    # Scenario 4: Forced truncation test with small context window
    print("--- Scenario 4: Forced truncation test (artificial 1000 token limit) ---")

    # Create new TestRagService instance with small context limit
    rag_small = TestRagService(mock_docs, max_context_tokens_override=1000)
    print("Initialized new TestRagService with max_context_tokens = 1000")

    # Create very long history to exceed 1000 tokens
    long_history = []
    long_text = "This is a very long message repeated many times to consume tokens. " * 50  # ~2500 chars

    for i in range(10):  # 10 user/assistant pairs
        long_history.extend([
            {"role": "user", "content": f"Question {i}: How does machine learning work? {long_text}"},
            {"role": "assistant", "content": f"Answer {i}: ML works through training on data and algorithms. {long_text}"}
        ])

    question4 = "Can you summarize ML approaches?"
    history4 = long_history[:]
    
    print(f"Input History messages: {len(history4)}")
    answer4, docs4 = rag_small.answer(question4, history4, debug=True)
    print(f"Q: {question4}")
    print(f"A: {answer4}\n")

    print("Simulation complete. Check logging output for internal pipeline details.")

if __name__ == "__main__":
    run_simulation()
