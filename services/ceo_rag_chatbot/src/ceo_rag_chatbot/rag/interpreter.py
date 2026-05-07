
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
import json
import logging
import re
from pydantic import BaseModel, Field, model_validator


class Interpretation(BaseModel):
    """Enhanced interpretation results with balanced metadata."""

    # Core question data
    original_question: str
    interpreted_question: str

    # Interpretation quality tracking
    interpreted_successfully: bool
    interpretation_confidence: float = Field(ge=0.0, le=1.0)  # How well did I understand?
    routing_confidence: float = Field(ge=0.0, le=1.0)        # How confident in routing decision?

    # Routing decision
    needs_retrieval: bool
    search_query: Optional[str] = None # if needs_retrieval=True, a search_query to be passed to vectordb retriever

    # Context and metadata
    context_used: str = Field(pattern=r"^(standalone|enhanced)$")  # "standalone" or "enhanced"
    reasoning: str

    # Optional rich metadata
    resolved_references: Optional[List[str]] = None      # What references were resolved
    suggested_improvements: Optional[List[str]] = None   # Better ways to ask

    # Validation: search_query only when retrieval needed
    @model_validator(mode="after")
    def validate_routing_logic(self) -> "Interpretation":
        if self.needs_retrieval:
            if self.search_query is None:
                raise ValueError("search_query required when needs_retrieval=True")
        else:
            if self.search_query is not None:
                raise ValueError("search_query should be None when needs_retrieval=False")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump()


class QuestionInterpreter:
    """
    Analyzes questions in conversation context to determine optimal response strategy.

    This class performs Phase 1 of the two-phase reasoning approach:
    1. Analyze question intent and conversational references
    2. Determine if retrieval is needed or direct answer suffices
    3. Generate appropriate search queries or direct responses
    """

    def __init__(self, llm_provider):
        """
        Initialize interpreter with shared LLM infrastructure.

        Args:
            llm_provider: LLMProvider instance (shared with RagService)
        """
        self.llm = llm_provider

    def interpret_question(self, question: str, history: List[Dict[str, str]]) -> Interpretation:
        """
        Main interpretation method - Phase 1 of two-phase reasoning.
        Now performs interpretation for ALL questions, regardless of history.

        Args:
            question: The user's current question
            history: List of previous conversation messages (optional for enhanced context)

        Returns:
            Interpretation: Structured analysis of question intent and strategy
        """
        try:
            # Build interpretation prompt (handles both standalone and contextual cases)
            prompt = self._build_interpretation_prompt(question, history)

            # Generate interpretation with optimized parameters
            kwargs = {
                # while this would be smart to do, cannot seem to set a safe upper token limit that does not cut off generated interpretation reponses especially when input prmprt is larger (long convos/questions)
                # "max_output_tokens":1024, # Optimized for JSON output - 
                "temperature":0.1, # deterministic
                "response_mime_type": "application/json" # ensure response is parseable JSON
                } 
            response = self.llm.generate(
                prompt,
                **kwargs     
            )

            # Parse and validate response
            interpretation = self._parse_interpretation_response(response, question, history)

            return interpretation

        except Exception as e:
            logging.warning(f"Warning: Question interpretation failed: {e}")
            # Return safe fallback interpretation
            return self._create_fallback_interpretation(question, history)

    def _build_interpretation_prompt(self, question: str, history: List[Dict[str, str]]) -> str:
        """
        Build the interpretation prompt that guides LLM analysis for both standalone and contextual questions.

        Args:
            question: Current user question
            history: Conversation history (may be empty for standalone questions)

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
        else:
            history_text = "No previous conversation history available."

        # Add context quality assessment
        context_quality = self._assess_context_quality_fast(history)
        question_intent = self._analyze_question_intent_fast(question)
        technical_domain = self._is_technical_domain_fast(question,history)
        prompt = f"""Analyze this question and determine the optimal response strategy. 
This analysis applies to both standalone questions (no history) and follow-up questions (with history).

CRITICAL: This is ONLY for analysis and routing. DO NOT generate answers. Answers will be generated separately.

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {question}

Your task is to understand what the user is actually asking and determine how to answer it effectively.

RESPONSE FORMAT: Return a valid JSON object with this exact structure:
```json
{{
    # Core question data
    "original_question": str, # the original question passed in
    "interpreted_question": str, # your interpretation of what the user is actually asking
    
    # Interpretation quality tracking
    "interpreted_successfully": bool, # True if interpretation was successful, False otherwise
    "interpretation_confidence": float, # How well did I understand?
    "routing_confidence": float, # How confident in routing decision?

    # Routing decision
    "needs_retrieval": bool, # True if retrieval is needed, False otherwise
    "search_query": Optional[str] = None, # if needs_retrieval=True, a search_query to be passed to vectordb retriever

    # Context and metadata
    "context_used": str = Field(pattern=r"^(standalone|enhanced)$"),  # "enhanced" if history is provided, else "standalone"
    "reasoning": str, # Explain your reasoning for routing decision

    # Optional rich metadata
    "resolved_references": Optional[List[str]] = None,      # What references were resolved
    "suggested_improvements": Optional[List[str]] = None   # Better ways to ask
}}
```

ANALYSIS GUIDELINES:

CONTEXT QUALITY ASSESSMENT METRICS:
- We have pre-calculated objective metrics for you based on simple keyword heuristics. 
- Use these 3 objective, deterministic context quality metrics to aid you in your interpretaion work,

1. context quality score: {context_quality}/10 (score out of 10 rating context quality based on keyword matches)
2. Question intent: {question_intent} (one of: ['detail-seeking','repetition','information-seeking']))
3. In Technical Domain: {technical_domain} (bool)

- Their role is described in the following rules sections that follow:

CRITICAL RULES (Apply these rules strictly):
- If question intent is 'detail-seeking' or 'information-seeking' → needs_retrieval MUST be True
- If context quality < 6 → needs_retrieval MUST be True
- When in doubt, prefer setting needs_retrieval to True (pure follow-ups are assumed to be minority case)
- DO NOT generate answers - only analyze and route

DETAIL-SEEKING DETECTION:
- Questions with 'more', 'details', 'explain', 'how', 'why', 'benefits', 'advantages', 'process', 'steps' → retrieval needed
- Questions about 'benefits', 'advantages', 'disadvantages' → retrieval needed
- Questions asking 'what is the process for' → retrieval needed

CONTEXT QUALITY ASSESSMENT:
- If history contains < 3 technical terms → retrieval needed
- If question asks for specific procedures → retrieval needed
- If history is generic/summary-only → retrieval needed

TECHNICAL DOMAINS (prefer retrieval):
- Questions about: sampling, plots, imagery, coordinates, analysis
- Domain keywords: CEO, Collect Earth, GIS, remote sensing

REPETITION QUESTIONS (no retrieval needed):
- Contains: repeat, again, what did you say, summarize
- Examples: "Can you repeat that?", "What did you just say?", "I don't understand."

JSON RESPONSE:"""

        return prompt

    def _parse_interpretation_response(self, response: str, original_question: str, history: List[Dict[str, str]]) -> Interpretation:
        """
        Parse LLM response into structured Interpretation object.

        details:
            Attempts to extract the most valid JSON object from the LLM response.
            Handles cases where the LLM might include duplicate responses, 
            Markdown code blocks (including unclosed ones), or extra explanatory text.

        Args:
            response: Raw LLM response (expected to be JSON or contain JSON)
            original_question: Original user question for fallback
            history: Conversation history for context

        Returns:
            Structured interpretation result
        """
        # try to load response right away as JSON
        try:
            data = json.loads(response)

            # Determine context used
            context_used = "enhanced" if history else "standalone"

            # Extract confidence (handle both single 'confidence' and split fields)
            interpretation_confidence = data.get("interpretation_confidence", data.get("confidence", 0.5))
            routing_confidence = data.get("routing_confidence", data.get("confidence", 0.5))

            # Validate required fields and provide defaults
            return Interpretation(
                original_question=original_question,
                interpreted_question=data.get("interpreted_question", original_question),
                interpreted_successfully=True,
                interpretation_confidence=interpretation_confidence,
                routing_confidence=routing_confidence,
                needs_retrieval=data.get("needs_retrieval", True),
                search_query=data.get("search_query"),
                context_used=context_used,
                reasoning=data.get("reasoning", "Analysis completed")
            )
        except:
            # go down complex string parsing rabbit hole to try to create parseable json
            # I had an LLM write this and I don't put much faith in it.
            # this was actually a solution to the 'unparseable response' problem I had to a HuggingFace model 
            # Gemini models return parseable json without issue.
            logging.warning("Warning: could not load LLM response as json, attempting to parse")
            try:
                json_candidates = []

                # 1. Look for JSON in Markdown code blocks first (including potentially unclosed ones)
                # Match from ```json until either closing ``` or end of string
                code_block_match = re.search(r"```(?:json)?\s*(\{.*)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
                if code_block_match:
                    json_candidates.append(code_block_match.group(1).strip())
                
                # 2. Extract all { ... } blocks and try parsing them from the end (prefer last valid JSON)
                potential_blocks = []
                start = 0
                while True:
                    idx = response.find('{', start)
                    if idx == -1:
                        break
                    
                    # Match braces to find full objects
                    brace_count = 0
                    found_end = False
                    for i in range(idx, len(response)):
                        if response[i] == '{':
                            brace_count += 1
                        elif response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                potential_blocks.append(response[idx:i+1])
                                start = i + 1
                                found_end = True
                                break
                    
                    if not found_end:
                        # If we didn't find a matching '}', this block might be truncated
                        potential_blocks.append(response[idx:])
                        start = idx + 1
                
                json_candidates.extend(reversed(potential_blocks))

                # 3. Try parsing candidates until one works
                data = None
                for candidate in json_candidates:
                    # Pre-process candidate to handle common LLM mistakes
                    # Strip Python/shell style comments (# comment)
                    cleaned = re.sub(r'#.*$', '', candidate, flags=re.MULTILINE)
                    
                    # Attempt to repair potentially truncated JSON
                    repaired = self._try_repair_json(cleaned)
                    
                    try:
                        parsed = json.loads(repaired)
                        # Basic validation: check for expected keys
                        if any(k in parsed for k in ["interpreted_question", "needs_retrieval", "search_query"]):
                            data = parsed
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue

                if data is None:
                    # Fallback to the largest { ... } block if all else fails
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1:
                        content = response[json_start:json_end] if json_end > json_start else response[json_start:]
                        cleaned_fallback = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
                        repaired_fallback = self._try_repair_json(cleaned_fallback)
                        data = json.loads(repaired_fallback)
                    else:
                        # Log the failed response for debugging
                        logging.warning(f"Warning: Failed to find JSON in response: {response}")
                        raise ValueError("No JSON found in response")

                # Determine context used
                context_used = "enhanced" if history else "standalone"

                # Extract confidence (handle both single 'confidence' and split fields)
                interpretation_confidence = data.get("interpretation_confidence", data.get("confidence", 0.5))
                routing_confidence = data.get("routing_confidence", data.get("confidence", 0.5))

                # Validate required fields and provide defaults
                return Interpretation(
                    original_question=original_question,
                    interpreted_question=data.get("interpreted_question", original_question),
                    interpreted_successfully=True,
                    interpretation_confidence=interpretation_confidence,
                    routing_confidence=routing_confidence,
                    needs_retrieval=data.get("needs_retrieval", True),
                    search_query=data.get("search_query"),
                    context_used=context_used,
                    reasoning=data.get("reasoning", "Analysis completed")
                )

            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logging.warning(f"Warning: Failed to parse interpretation response: {e}")
                return self._create_fallback_interpretation(original_question, history)

    def _create_fallback_interpretation(self, question: str, history: List[Dict[str, str]]) -> Interpretation:
        """
        Create safe fallback interpretation when analysis fails.

        Args:
            question: Original question to use as fallback
            history: Conversation history for context

        Returns:
            Conservative interpretation that assumes retrieval is needed
        """
        context_used = "enhanced" if history else "standalone"

        return Interpretation(
            original_question=question,
            interpreted_question=question,
            interpreted_successfully=False,  # Interpretation failed
            interpretation_confidence=0.1,   # Low confidence due to failure
            routing_confidence=0.5,          # Neutral confidence in fallback routing
            needs_retrieval=True,            # Safe fallback: assume retrieval needed
            search_query=question,
            context_used=context_used,
            reasoning="Fallback: assuming retrieval needed due to interpretation failure"
        )

    def _analyze_question_intent_fast(self, question: str) -> str:
        """Fast keyword-based question intent analysis."""
        question_lower = question.lower()
        
        detail_keywords = ['more', 'details', 'explain', 'how', 'why', 'benefits', 'advantages', 'process', 'steps', 'compare']
        repetition_keywords = ['repeat', 'again', 'what did you say', 'summarize']
        
        if any(keyword in question_lower for keyword in detail_keywords):
            intent = 'detail-seeking'
        
        elif any(keyword in question_lower for keyword in repetition_keywords):
            intent = 'repetition'
        
        # default to retrieval with a catch-all intent value
        else:
            intent = 'information-seeking'
        
        return intent
    
    def _assess_context_quality_fast(self, history: List[Dict[str, str]]) -> float:
        """Fast heuristic-based context quality assessment."""
        if not history:
            return 0.0
        
        # Count technical terms in history
        technical_terms = ['samples','sampling', 'plot','design', 
                           'imagery','image', 'collection','polygons','source','data', 
                           'coordinates', 'resolution', 'scale','reference',
                           'classification', 'landcover', 'land cover','analysis','area','estimation',
                           'forest','inventory']
        history_text = ' '.join([msg.get('content', '') for msg in history]).lower()
        
        technical_score = sum(1 for term in technical_terms if term in history_text)
        
        # Check if history contains specific details vs. generic responses
        detail_indicators = ['specifically', 'exactly', 'precisely', 'step', 'procedure', 'method']
        detail_score = sum(1 for indicator in detail_indicators if indicator in history_text)
        
        # Simple scoring: 0-10 scale
        # technical terms are double-weighted compared to detail terms
        quality_score = min(10, (technical_score * 2) + (detail_score * 1))
        return quality_score

    def _is_technical_domain_fast(self, question: str, history: List[Dict[str, str]]) -> bool:
        """Fast domain detection for technical content."""
        technical_domains = ['ceo', 'collect earth', 'gis', 'remote sensing', 'sampling', 'plot', 'imagery']
        combined_text = (question + ' ' + ' '.join([msg.get('content', '') for msg in history])).lower()
        
        return any(domain in combined_text for domain in technical_domains)

    def _try_repair_json(self, json_str: str) -> str:
        """
        Attempt to repair truncated JSON by appending missing quotes and braces.
        
        Args:
            json_str: Potentially truncated JSON string
            
        Returns:
            The original string if valid, or a repaired version if possible.
        """
        # Remove trailing junk that might prevent simple brace appending
        json_str = json_str.strip()
        
        # If it already looks valid, don't touch it
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass

        # Try common repair strategies
        strategies = [
            lambda s: s + '"',           # Close a string
            lambda s: s + '"}',          # Close a string and an object
            lambda s: s + '"]}',         # Close a string, an array, and an object
            lambda s: s + '}',           # Close an object
            lambda s: s + ']}',          # Close an array and an object
            lambda s: s + '"' + '}' * 2, # Close string and two objects
            lambda s: s + '"' + '}' * 3, # Close string and three objects
        ]

        for strategy in strategies:
            try:
                test_str = strategy(json_str)
                json.loads(test_str)
                return test_str
            except json.JSONDecodeError:
                continue
        
        # Fallback to adding multiple braces
        for i in range(1, 5):
            try:
                test_str = json_str + ("}" * i)
                json.loads(test_str)
                return test_str
            except json.JSONDecodeError:
                continue
                
        return json_str
