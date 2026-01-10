"""
 #! src/ceo_chatbot/rag/interpreter.py
 summary:
    Question interpretation engine for conversational RAG.

 details:
    Analyzes user questions in the context of conversation history to determine
    intent, reference resolution, and optimal response strategy. Enables intelligent
    routing between conversational follow-ups and document-based answers.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import logging


@dataclass
class Interpretation:
    """Structured representation of question analysis results."""
    interpreted_question: str
    is_followup: bool
    needs_retrieval: bool
    search_query: Optional[str]
    direct_answer: Optional[str]
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "interpreted_question": self.interpreted_question,
            "is_followup": self.is_followup,
            "needs_retrieval": self.needs_retrieval,
            "search_query": self.search_query,
            "direct_answer": self.direct_answer,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class QuestionInterpreter:
    """
    Analyzes questions in conversation context to determine optimal response strategy.

    This class performs Phase 1 of the two-phase reasoning approach:
    1. Analyze question intent and conversational references
    2. Determine if retrieval is needed or direct answer suffices
    3. Generate appropriate search queries or direct responses
    """

    def __init__(self, llm_pipeline, tokenizer):
        """
        Initialize interpreter with shared LLM infrastructure.

        Args:
            llm_pipeline: HuggingFace text generation pipeline (shared with RagService)
            tokenizer: Associated tokenizer for the LLM
        """
        self.llm = llm_pipeline
        self.tokenizer = tokenizer

    def interpret_question(self, question: str, history: List[Dict[str, str]]) -> Interpretation:
        """
        Main interpretation method - Phase 1 of two-phase reasoning.

        Args:
            question: The user's current question
            history: List of previous conversation messages

        Returns:
            Interpretation: Structured analysis of question intent and strategy
        """
        # For first questions with no history, skip interpretation and assume retrieval needed
        if not history:
            return Interpretation(
                interpreted_question=question,
                is_followup=False,
                needs_retrieval=True,  # First questions typically need retrieval
                search_query=question,
                direct_answer=None,
                confidence=1.0,  # High confidence for safe defaults
                reasoning="First question with no history - assuming retrieval needed"
            )

        # For follow-up questions, perform full interpretation
        try:
            # Build interpretation prompt
            prompt = self._build_interpretation_prompt(question, history)

            # Generate interpretation with optimized parameters
            response = self.llm(
                prompt,
                max_new_tokens=300,  # Structured responses
                temperature=0.1,     # Deterministic
                do_sample=True,      # But low creativity
                return_full_text=False
            )[0]["generated_text"]

            # Parse and validate response
            interpretation = self._parse_interpretation_response(response, question)

            return interpretation

        except Exception as e:
            logging.warning(f"Question interpretation failed: {e}")
            # Return safe fallback interpretation
            return self._create_fallback_interpretation(question)

    def _build_interpretation_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Build the interpretation prompt that guides LLM analysis.

        Args:
            question: Current user question
            history: Conversation history

        Returns:
            Formatted prompt for interpretation
        """
        # Format history for context
        history_text = ""
        if history:
            history_entries = []
            for msg in history[-10:]:  # Limit to recent history
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]  # Truncate long messages
                history_entries.append(f"{role}: {content}")
            history_text = "\n".join(history_entries)

        prompt = f"""Analyze this question in the context of the conversation history and determine the optimal response strategy.

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {question}

Your task is to understand what the user is actually asking and determine how to answer it effectively.

RESPONSE FORMAT: Return a valid JSON object with this exact structure:
{{
    "interpreted_question": "Restate what the user is actually asking based on context",
    "is_followup": true/false,
    "needs_retrieval": true/false,
    "search_query": "specific search terms if retrieval needed, null otherwise",
    "direct_answer": "complete answer if no retrieval needed, null otherwise",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of your analysis"
}}

ANALYSIS GUIDELINES:
- If question contains pronouns/references ("it", "this", "step 7"), check history to resolve them
- Pure follow-ups about previous discussion: set needs_retrieval=false, provide direct_answer
- Questions needing external knowledge: set needs_retrieval=true, provide focused search_query
- Ambiguous questions: prefer interpretation over asking for clarification

JSON RESPONSE:"""

        return prompt

    def _parse_interpretation_response(self, response: str, original_question: str) -> Interpretation:
        """
        Parse LLM response into structured Interpretation object.

        Args:
            response: Raw LLM response (expected to be JSON)
            original_question: Fallback question if parsing fails

        Returns:
            Structured interpretation result
        """
        try:
            # Extract JSON from response (LLM might add extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Validate required fields and provide defaults
            return Interpretation(
                interpreted_question=data.get("interpreted_question", original_question),
                is_followup=data.get("is_followup", False),
                needs_retrieval=data.get("needs_retrieval", True),
                search_query=data.get("search_query"),
                direct_answer=data.get("direct_answer"),
                confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),  # Clamp to 0-1
                reasoning=data.get("reasoning", "Analysis completed")
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"Failed to parse interpretation response: {e}")
            return self._create_fallback_interpretation(original_question)

    def _create_fallback_interpretation(self, question: str) -> Interpretation:
        """
        Create safe fallback interpretation when analysis fails.

        Args:
            question: Original question to use as fallback

        Returns:
            Conservative interpretation that assumes retrieval is needed
        """
        return Interpretation(
            interpreted_question=question,
            is_followup=False,
            needs_retrieval=True,
            search_query=question,
            direct_answer=None,
            confidence=0.5,
            reasoning="Fallback: assuming retrieval needed due to interpretation failure"
        )

    def _is_conversational_followup(self, question: str, history: List[Dict[str, str]]) -> bool:
        """
        Heuristic check for conversational follow-up patterns.

        Args:
            question: Question to analyze
            history: Conversation history

        Returns:
            True if question appears to be a follow-up
        """
        if not history:
            return False

        followup_indicators = [
            # Pronouns
            " it ", " this ", " that ", " those ", " them ",

            # Follow-up phrases
            "how do i", "what about", "tell me more", "explain",

            # References to previous content
            "step ", "option ", "number ", "the one",

            # Question words that often reference history
            "why", "how come", "what do you mean"
        ]

        question_lower = question.lower()
        return any(indicator in question_lower for indicator in followup_indicators)