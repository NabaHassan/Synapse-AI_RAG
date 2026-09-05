"""
Query Analysis & Classification - Step 2 of Query Processing
Classifies queries into types and calculates complexity/relevance scores
Uses zero-shot classification with cross-encoder/nli-deberta-v3-base
"""

import re
import spacy
import torch
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """
    Classification result for a query
    NOTE: routing_decision removed - use QueryRouter for routing
    """
    query_text: str
    query_type: str  # factual, reasoning, conversational, hybrid, out_of_scope
    confidence: float  # 0.0-1.0
    complexity: float  # 0.0-1.0
    requires_long_response: bool  # Whether query needs comprehensive answer (for min_tokens)
    type_scores: Dict[str, float]  # Scores for all types
    metadata: Dict[str, Any]  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ComplexityAnalyzer:
    """
    Analyzes query complexity using various heuristics
    """

    def __init__(self, use_spacy: bool = True):
        """
        Initialize complexity analyzer
        
        Args:
            use_spacy: Whether to use spaCy for entity extraction
        """
        self.use_spacy = use_spacy
        self.nlp = None

        if use_spacy and spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded for complexity analysis")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False

        # Complexity indicators
        self.comparative_words = {
            'compare', 'contrast', 'difference', 'versus', 'vs', 'better', 'worse',
            'more', 'less', 'similar', 'different', 'analyze', 'evaluate'
        }

        self.analytical_words = {
            'why', 'how', 'explain', 'analyze', 'evaluate', 'assess', 'determine',
            'investigate', 'explore', 'understand', 'reason', 'cause', 'effect'
        }

        self.multi_hop_indicators = {
            'and then', 'after that', 'following', 'subsequently', 'therefore',
            'as a result', 'consequently', 'in addition', 'furthermore'
        }

    def calculate_complexity(self, query_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate query complexity score (0.0-1.0)
        
        Args:
            query_text: Query text
            
        Returns:
            Tuple of (complexity_score, metadata)
        """
        metadata = {}
        factors = []

        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        # Factor 1: Query length (longer = potentially more complex)
        word_count = len(query_text.split())
        length_score = min(word_count / 20.0, 1.0)  # Normalize to 20 words
        factors.append(length_score * 0.2)
        metadata['word_count'] = word_count

        # Factor 2: Number of entities (if spaCy available)
        entity_count = 0
        if self.nlp:
            try:
                doc = self.nlp(query_text)
                entity_count = len(doc.ents)
                entity_score = min(entity_count / 3.0, 1.0)  # Normalize to 3 entities
                factors.append(entity_score * 0.2)
                metadata['entity_count'] = entity_count
                metadata['entities'] = [ent.text for ent in doc.ents]
            except Exception as e:
                logger.debug(f"Entity extraction failed: {e}")

        # Factor 3: Comparative/analytical words
        comparative_count = len(query_words & self.comparative_words)
        analytical_count = len(query_words & self.analytical_words)
        analytical_score = min((comparative_count + analytical_count) / 3.0, 1.0)
        factors.append(analytical_score * 0.3)
        metadata['comparative_words'] = comparative_count
        metadata['analytical_words'] = analytical_count

        # Factor 4: Multi-hop indicators
        multi_hop_count = sum(1 for phrase in self.multi_hop_indicators if phrase in query_lower)
        multi_hop_score = min(multi_hop_count / 2.0, 1.0)
        factors.append(multi_hop_score * 0.15)
        metadata['multi_hop_indicators'] = multi_hop_count

        # Factor 5: Number of questions/clauses
        question_marks = query_text.count('?')
        clause_count = query_text.count(',') + query_text.count(';')
        structure_score = min((question_marks + clause_count) / 3.0, 1.0)
        factors.append(structure_score * 0.15)
        metadata['question_marks'] = question_marks
        metadata['clause_count'] = clause_count

        # Calculate final complexity
        if factors:
            complexity = sum(factors)
        else:
            complexity = 0.3  # Default moderate complexity

        # Clamp to 0.1-0.95 range (avoid extremes)
        complexity = max(0.1, min(complexity, 0.95))

        metadata['complexity_factors'] = {
            'length': length_score * 0.2,
            'analytical': analytical_score * 0.3,
            'multi_hop': multi_hop_score * 0.15,
            'structure': structure_score * 0.15
        }

        return complexity, metadata

    def is_complex_query(self, query_text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if query requires long-form response (for min_tokens decision).
        
        This method analyzes the actual user query semantically to detect if it requires
        a comprehensive answer, rather than relying on prompt length which grows with
        conversational context.
        
        Args:
            query_text: User's query text
            
        Returns:
            Tuple of (requires_long_response, metadata with reasoning)
        """
        metadata = {}
        complexity_indicators = []

        query_lower = query_text.lower()
        query_words = set(query_lower.split())

        # Indicator 1: Multiple questions in one query
        question_marks = query_text.count('?')
        if question_marks > 1:
            complexity_indicators.append('multiple_questions')
            metadata['question_count'] = question_marks

        # Indicator 2: Comparative language
        comparative_keywords = {
            'compare', 'contrast', 'difference', 'differences', 'versus', 'vs',
            'vs.', 'better', 'worse', 'similar', 'different'
        }
        has_comparative = bool(query_words & comparative_keywords)
        if has_comparative:
            complexity_indicators.append('comparative_language')
            metadata['comparative_words'] = list(query_words & comparative_keywords)

        # Indicator 3: Analytical requests
        analytical_keywords = {
            'analyze', 'analysis', 'evaluate', 'assessment', 'assess',
            'explain why', 'implications', 'impact', 'effects',
            'comprehensive', 'detailed', 'thorough', 'thoroughly',
            'breakdown', 'break down'
        }
        # Check both individual words and phrases
        has_analytical = bool(query_words & analytical_keywords)
        analytical_phrases = ['explain why', 'break down', 'in detail']
        has_analytical_phrase = any(phrase in query_lower for phrase in analytical_phrases)

        if has_analytical or has_analytical_phrase:
            complexity_indicators.append('analytical_request')
            metadata['analytical_detected'] = True

        # Indicator 4: Multi-step reasoning
        multi_step_keywords = {
            'first', 'then', 'after that', 'following', 'subsequently',
            'therefore', 'as a result', 'consequently', 'next'
        }
        multi_step_phrases = [
            'first...then', 'if...then', 'if...how', 'step by step',
            'one by one', 'in order'
        ]
        has_multi_step = bool(query_words & multi_step_keywords)
        has_multi_step_phrase = any(phrase in query_lower for phrase in multi_step_phrases)

        if has_multi_step or has_multi_step_phrase:
            complexity_indicators.append('multi_step_reasoning')
            metadata['multi_step_detected'] = True

        # Indicator 5: Requests for comprehensive answers
        comprehensive_keywords = {
            'all', 'every', 'everything', 'complete', 'entire', 'full',
            'comprehensive', 'exhaustive', 'total'
        }
        comprehensive_phrases = [
            'all of', 'all the', 'every aspect', 'complete list',
            'full analysis', 'in full', 'everything about'
        ]
        has_comprehensive = bool(query_words & comprehensive_keywords)
        has_comprehensive_phrase = any(phrase in query_lower for phrase in comprehensive_phrases)

        if has_comprehensive or has_comprehensive_phrase:
            complexity_indicators.append('comprehensive_request')
            metadata['comprehensive_detected'] = True

        # Indicator 6: Legal-specific complex queries
        legal_complex_keywords = {
            'points and authorities', 'legal issues', 'legal analysis',
            'case law', 'statutory interpretation', 'precedent'
        }
        has_legal_complex = any(keyword in query_lower for keyword in legal_complex_keywords)

        if has_legal_complex:
            complexity_indicators.append('legal_complex_query')
            metadata['legal_complex_detected'] = True

        # Simple query patterns (should NOT trigger long response)
        simple_patterns = [
            r'^what is\s+\w+\s*\?*$',  # "What is X?"
            r'^who (is|was|created|made)\s+',  # "Who is/created X?"
            r'^when (did|was)\s+',  # "When did/was X?"
            r'^where (is|was)\s+',  # "Where is/was X?"
            r'^define\s+\w+\s*\?*$',  # "Define X?"
            r'^(yes|no)\s*\?*$',  # Yes/no questions
        ]

        is_simple_pattern = any(re.match(pattern, query_lower) for pattern in simple_patterns)

        # Short follow-up patterns (conversational references)
        followup_patterns = [
            r'^(what|how) about (that|this|it)\s*\?*$',
            r'^and (that|this|it)\s*\?*$',
            r'^(tell me )?(more|else)\s*\?*$',
        ]

        is_followup = any(re.match(pattern, query_lower) for pattern in followup_patterns)

        # Decision logic
        requires_long_response = False

        if is_simple_pattern or is_followup:
            # Explicitly simple queries
            requires_long_response = False
            metadata['decision_reason'] = 'simple_pattern_match'
        elif len(complexity_indicators) >= 2:
            # Multiple complexity indicators = definitely complex
            requires_long_response = True
            metadata['decision_reason'] = 'multiple_complexity_indicators'
        elif len(complexity_indicators) == 1:
            # Single strong indicator = complex
            requires_long_response = True
            metadata['decision_reason'] = 'single_complexity_indicator'
        else:
            # No indicators = simple query
            requires_long_response = False
            metadata['decision_reason'] = 'no_complexity_indicators'

        metadata['complexity_indicators'] = complexity_indicators
        metadata['indicator_count'] = len(complexity_indicators)

        return requires_long_response, metadata


class GenericQueryDetector:
    """
    Detects if a query is too generic/vague to be meaningful
    Used to filter out non-informative queries
    """

    def __init__(self):
        """
        Initialize generic query detector
        """
        logger.info("GenericQueryDetector initialized")

    @staticmethod
    def is_generic(query_text: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if query is too generic/vague to be meaningful
        
        Args:
            query_text: Query text
            
        Returns:
            Tuple of (is_generic, metadata)
        """
        metadata = {}

        # Empty or very short
        if not query_text or len(query_text.strip()) < 3:
            metadata['reason'] = 'too_short'
            return True, metadata

        query_words = query_text.lower().split()
        word_count = len(query_words)

        # Stop words
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an',
            'it', 'this', 'that', 'tell', 'me', 'about', 'of', 'in', 'to', 'for', 'on'
        }

        content_words = [w for w in query_words if w not in stop_words]
        content_word_count = len(content_words)

        # Pure generic queries like "What is it?" or "Tell me about that"
        if content_word_count == 0:
            metadata['reason'] = 'no_content_words'
            metadata['content_word_count'] = 0
            return True, metadata

        # Very short queries with minimal content (1 word queries like "What is X?")
        if word_count <= 3 and content_word_count <= 1:
            metadata['reason'] = 'too_vague'
            metadata['content_word_count'] = content_word_count
            return True, metadata

        # Not generic
        metadata['content_word_count'] = content_word_count
        metadata['total_word_count'] = word_count
        return False, metadata


class QueryClassifier:
    """
    Main query classifier using zero-shot classification
    Uses cross-encoder/nli-deberta-v3-base for NLI-based classification
    """

    INFORMATION_REQUEST_RE = re.compile(
        r"^\s*(?:"
        r"what|who|when|where|why|which|how|"
        r"did|does|do|is|are|was|were|"
        r"tell me|show me|list|find|identify|explain|describe|summarize|summarise"
        r")\b",
        re.IGNORECASE,
    )
    SPECIFIC_ANCHOR_RE = re.compile(
        r"\b(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
        r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"
        r")\b"
    )
    UNSAFE_INSTRUCTION_RE = re.compile(
        r"\b(?:how\s+(?:do|can)\s+i|instructions?|steps?|guide)\b.{0,80}"
        r"\b(?:kill|harm|poison|make\s+a\s+bomb|build\s+a\s+bomb|explosive|hack\s+into|phish|steal|"
        r"commit\s+fraud|launder|blackmail|extort|doxx|stalk|malware)\b",
        re.IGNORECASE,
    )

    def __init__(
            self,
            model_name: str = "cross-encoder/nli-deberta-v3-base",
            device: str = "cuda" if torch and torch.cuda.is_available() else "cpu",
            use_spacy: bool = True,
            classification_threshold: float = 0.5
    ):
        """
        Initialize query classifier
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
            device: Device to run model on ('cpu' or 'cuda')
            use_spacy: Whether to use spaCy for complexity analysis
            classification_threshold: Minimum confidence threshold for classification
        """
        self.model_name = model_name
        self.device = device
        self.classification_threshold = classification_threshold

        # Query type definitions
        self.query_types = {
            'factual': "This query asks for factual information or specific data that can be found in documents.",
            'reasoning': "This query requires reasoning, analysis, or inference beyond simple fact lookup.",
            'conversational': "This is a conversational query like greetings, small talk, or general conversation.",
            'hybrid': "This query combines factual information needs with reasoning or analysis.",
            'out_of_scope': "This query is unrelated to the domain or inappropriate."
        }

        # Initialize zero-shot classifier
        self.classifier = None
        self.tokenizer = None
        self.model = None

        if AutoTokenizer and AutoModelForSequenceClassification:
            try:
                logger.info(f"Loading zero-shot classification model: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

                if torch and device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to(device)
                    logger.info("Using GPU for classification")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for classification")

                self.model.eval()
                logger.info(" Classification model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load classification model: {e}")
                logger.warning("Falling back to rule-based classification")
        else:
            logger.warning("Transformers not available. Using rule-based classification only.")

        # Initialize complexity analyzer
        self.complexity_analyzer = ComplexityAnalyzer(use_spacy=use_spacy)

        # Initialize generic query detector
        self.generic_detector = GenericQueryDetector()

        # Rule-based patterns for fallback
        self.rule_patterns = {
            'conversational': [
                r'^(hi|hello|hey|good morning|good afternoon|good evening)',
                r'^(thanks|thank you|bye|goodbye)',
                r'^(how are you|what\'?s up)'
            ],
            'factual': [
                r'^(what is|what are|define|explain)',
                r'^(when|where|who)',
                r'(definition of|meaning of)'
            ],
            'reasoning': [
                r'^(why|how come)',
                r'(analyze|compare|evaluate|assess)',
                r'(reason|cause|effect|impact)'
            ]
        }

        logger.info("=" * 80)
        logger.info("QueryClassifier initialized")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Query types: {list(self.query_types.keys())}")
        logger.info(f"Classification threshold: {classification_threshold}")
        logger.info("=" * 80)

    def _classify_with_nli(self, query_text: str) -> Dict[str, float]:
        """
        Classify query using NLI-based zero-shot classification
        
        Args:
            query_text: Query text to classify
            
        Returns:
            Dictionary of query type -> confidence score
        """
        if not self.model or not self.tokenizer:
            return {}

        type_scores = {}

        for query_type, hypothesis in self.query_types.items():
            try:
                # Create premise (query) and hypothesis (type definition)
                inputs = self.tokenizer(
                    query_text,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                if self.device == "cuda" and torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get prediction
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=1)

                    # Get entailment score (typically index 2 for NLI models)
                    # [contradiction, neutral, entailment]
                    entailment_score = probs[0][2].item() if probs.shape[1] > 2 else probs[0][1].item()

                    type_scores[query_type] = entailment_score

            except Exception as e:
                logger.debug(f"NLI classification failed for {query_type}: {e}")
                type_scores[query_type] = 0.0

        return type_scores

    def _classify_with_rules(self, query_text: str) -> Dict[str, float]:
        """
        Fallback rule-based classification
        
        Args:
            query_text: Query text
            
        Returns:
            Dictionary of query type -> confidence score
        """
        type_scores = {qt: 0.0 for qt in self.query_types.keys()}
        query_lower = query_text.lower()

        # Check patterns
        for query_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    type_scores[query_type] = max(type_scores[query_type], 0.6)

        # Default to factual if no match
        if all(score < 0.3 for score in type_scores.values()):
            type_scores['factual'] = 0.5

        return type_scores

    @staticmethod
    def _select_knowledge_query_type(query_text: str, type_scores: Dict[str, float]) -> str:
        """
        Select the best retrieval-backed type for a concrete information request.
        """
        query_lower = query_text.lower().strip()

        if re.match(r"^(why|how come)\b", query_lower):
            return "reasoning"

        if re.search(r"\b(analy[sz]e|compare|evaluate|assess|reason|cause|effect|impact)\b", query_lower):
            return "reasoning"

        candidates = ("factual", "reasoning", "hybrid")
        best = max(candidates, key=lambda query_type: type_scores.get(query_type, 0.0))
        if type_scores.get(best, 0.0) > 0:
            return best

        if re.match(r"^(what|who|when|where|which|did|does|do|is|are|was|were|list|find|identify|show me)\b",
                    query_lower):
            return "factual"

        return "reasoning" if query_lower.startswith("how ") else "factual"

    @classmethod
    def _knowledge_request_metadata(cls, query_text: str, is_generic: bool) -> Dict[str, Any]:
        """
        Detect concrete information requests that should reach retrieval.

        The NLI classifier does not receive KB-domain context, so its out-of-scope
        label can be overconfident for document-grounded questions with dates,
        names, events, or other anchors.
        """
        words = re.findall(r"[A-Za-z0-9']+", query_text.lower())
        stop_words = {
            "what", "how", "why", "when", "where", "who", "which", "is", "are",
            "was", "were", "did", "do", "does", "the", "a", "an", "it", "this",
            "that", "tell", "me", "about", "of", "in", "to", "for", "on", "and",
            "or", "by", "with", "from", "as", "at",
        }
        content_words = [word for word in words if word not in stop_words]
        has_question_or_intent = bool(cls.INFORMATION_REQUEST_RE.search(query_text)) or "?" in query_text
        has_specific_anchor = bool(cls.SPECIFIC_ANCHOR_RE.search(query_text))
        has_enough_content = len(content_words) >= 3
        unsafe_instruction = bool(cls.UNSAFE_INSTRUCTION_RE.search(query_text))

        return {
            "has_question_or_intent": has_question_or_intent,
            "has_specific_anchor": has_specific_anchor,
            "content_word_count": len(content_words),
            "unsafe_instruction": unsafe_instruction,
            "is_knowledge_request": (
                    not is_generic
                    and has_question_or_intent
                    and (has_specific_anchor or has_enough_content)
                    and not unsafe_instruction
            ),
        }

    @classmethod
    def _apply_out_of_scope_guardrail(
            cls,
            query_text: str,
            query_type: str,
            confidence: float,
            type_scores: Dict[str, float],
            is_generic: bool
    ) -> Tuple[str, float, Dict[str, float], Dict[str, Any]]:
        """
        Re-route concrete document questions away from pre-retrieval rejection.
        """
        guardrail_metadata = cls._knowledge_request_metadata(query_text, is_generic)
        guardrail_metadata["applied"] = False

        if query_type != "out_of_scope" or not guardrail_metadata["is_knowledge_request"]:
            return query_type, confidence, type_scores, guardrail_metadata

        adjusted_type = cls._select_knowledge_query_type(query_text, type_scores)
        adjusted_confidence = max(type_scores.get(adjusted_type, 0.0), 0.55)
        adjusted_scores = dict(type_scores)
        adjusted_scores[adjusted_type] = adjusted_confidence
        adjusted_scores["out_of_scope"] = min(
            adjusted_scores.get("out_of_scope", 0.0),
            max(0.0, adjusted_confidence - 0.01)
        )

        guardrail_metadata.update({
            "applied": True,
            "original_type": query_type,
            "original_confidence": confidence,
            "adjusted_type": adjusted_type,
            "adjusted_confidence": adjusted_confidence,
            "raw_type_scores": dict(type_scores),
        })

        return adjusted_type, adjusted_confidence, adjusted_scores, guardrail_metadata

    def classify(self, query_text: str) -> QueryClassification:
        """
        Classify a query and calculate complexity/relevance scores
        
        Args:
            query_text: Query text to classify
            
        Returns:
            QueryClassification object with full analysis
        """
        logger.info(f"Classifying query: {query_text[:50]}...")

        # Step 1: Classify query type
        if self.model:
            type_scores = self._classify_with_nli(query_text)
        else:
            type_scores = self._classify_with_rules(query_text)

        # Get top type and confidence
        if type_scores:
            query_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[query_type]
        else:
            # Fallback
            type_scores = self._classify_with_rules(query_text)
            query_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[query_type]

        # Step 2: Calculate complexity
        complexity, complexity_metadata = self.complexity_analyzer.calculate_complexity(query_text)

        # Step 3: Detect if query is too generic
        is_generic, generic_metadata = self.generic_detector.is_generic(query_text)

        # Step 4: Determine if query requires long-form response (for min_tokens)
        requires_long_response, long_response_metadata = self.complexity_analyzer.is_complex_query(query_text)

        # Step 5: Prevent overconfident out-of-scope labels from rejecting
        # concrete, document-seeking questions before retrieval can run.
        query_type, confidence, type_scores, out_of_scope_guardrail = self._apply_out_of_scope_guardrail(
            query_text=query_text,
            query_type=query_type,
            confidence=confidence,
            type_scores=type_scores,
            is_generic=is_generic
        )

        # Combine metadata
        metadata = {
            'complexity_metadata': complexity_metadata,
            'generic_detection': generic_metadata,
            'is_generic': is_generic,
            'long_response_detection': long_response_metadata,
            'out_of_scope_guardrail': out_of_scope_guardrail,
            'classification_method': 'nli' if self.model else 'rules'
        }
        if out_of_scope_guardrail.get("applied"):
            metadata['classification_method'] = f"{metadata['classification_method']}_with_scope_guardrail"
            logger.info(
                "Out-of-scope guardrail applied: %s %.2f -> %s %.2f",
                out_of_scope_guardrail["original_type"],
                out_of_scope_guardrail["original_confidence"],
                out_of_scope_guardrail["adjusted_type"],
                out_of_scope_guardrail["adjusted_confidence"],
            )

        # Create classification result
        # Note: Routing is now handled by QueryRouter (separation of concerns)
        classification = QueryClassification(
            query_text=query_text,
            query_type=query_type,
            confidence=confidence,
            complexity=complexity,
            requires_long_response=requires_long_response,
            type_scores=type_scores,
            metadata=metadata
        )

        logger.info(f" Classification complete:")
        logger.info(f"   Type: {query_type} (confidence: {confidence:.2f})")
        logger.info(f"   Complexity: {complexity:.2f}")
        logger.info(f"   Requires long response: {requires_long_response}")
        logger.info(f"   Generic: {is_generic}")

        return classification

    def detect_continuation(
            self,
            query: str,
            last_turn: Optional[Any] = None,
            embedder: Optional[Any] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if query is a continuation of previous conversation.
        
        Uses two-stage detection:
        1. Pattern matching (fast, <1ms) - checks for continuation indicators
        2. Semantic similarity via BGE embeddings (optional, ~10-15ms) - validates topic similarity
        
        Args:
            query: Current query text
            last_turn: Previous conversation turn (optional)
            embedder: BGE embedder instance for semantic similarity (optional, reused from dense retriever)
            
        Returns:
            Tuple of (is_continuation, confidence, metadata)
            - is_continuation: True if query continues previous conversation
            - confidence: 0.0-1.0 confidence score
            - metadata: Dict with detection details
        
        Examples:
            >>> # After "Who is X?"
            >>> detect_continuation("What does that mean?", last_turn, embedder)
            (True, 0.85, {'reason': 'pattern_and_similarity', 'semantic_similarity': 0.42})
            
            >>> # After "Who is X?"
            >>> detect_continuation("Tell me about California law", last_turn, embedder)
            (False, 0.0, {'reason': 'no_continuation_pattern'})
        """
        metadata = {}

        # Quick pattern check first (no overhead)
        continuation_patterns = [
            r'\b(that|this|it|they|them|he|she|his|her|their)\b',  # Pronouns
            r'^(what|who|when|where|why|how) (does|is|was|were|did|do)',  # "what does that mean"
            r'^(tell me|explain|elaborate|describe)',  # "tell me more"
            r'^(and |also |what about|how about)',  # "and what else"
            r'^(more|else|continue)\s*\?*$',  # "more?", "else?"
            r'(mean|refer|talking about)\?*$',  # "what does that mean?"
        ]

        query_lower = query.lower().strip()

        # Check for continuation patterns
        has_pattern = any(re.search(p, query_lower) for p in continuation_patterns)

        if not has_pattern:
            metadata['reason'] = 'no_continuation_pattern'
            metadata['patterns_checked'] = len(continuation_patterns)
            return False, 0.0, metadata

        metadata['pattern_detected'] = True

        # If pattern detected but no last turn, assume continuation with medium confidence
        if not last_turn:
            metadata['reason'] = 'pattern_match_no_history'
            return True, 0.6, metadata

        # If we have embedder, validate with semantic similarity
        if embedder and hasattr(last_turn, 'query'):
            try:
                import numpy as np

                # Embed both queries (reuse BGE model!)
                query_emb = embedder.encode([query])[0]
                last_query_emb = embedder.encode([last_turn.query])[0]

                # Cosine similarity
                similarity = float(np.dot(query_emb, last_query_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(last_query_emb)
                ))

                metadata['semantic_similarity'] = similarity
                metadata['last_query'] = last_turn.query[:50] + '...' if len(last_turn.query) > 50 else last_turn.query

                # High similarity = likely same topic continuation
                # Threshold of 0.3 is conservative (allows topic drift while maintaining relevance)
                if similarity > 0.3:
                    metadata['reason'] = 'pattern_and_similarity'
                    # Confidence is weighted average of pattern match (0.7) and similarity
                    confidence = 0.3 * 0.7 + 0.7 * similarity
                    return True, confidence, metadata
                else:
                    # Pattern detected but low similarity = likely topic change
                    metadata['reason'] = 'pattern_but_low_similarity'
                    return False, similarity, metadata

            except Exception as e:
                logger.debug(f"Semantic similarity computation failed: {e}")
                metadata['similarity_error'] = str(e)

        # NEW: Check for explicit topic change
        # If query has a clear new subject (proper nouns, "Who is X", "What is X"), it's likely a topic change
        topic_change_patterns = [
            r'^who (is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Who is Jeremy Salvador"
            r'^what (is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "What is XYZ"
            r'^tell me about\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # "Tell me about X"
        ]

        for pattern in topic_change_patterns:
            match = re.search(pattern, query)
            if match:
                # Extract the new subject
                new_subject = match.group(2) if match.lastindex >= 2 else match.group(1)
                metadata['new_subject_detected'] = new_subject
                metadata['reason'] = 'explicit_topic_change'
                logger.debug(f"Detected explicit topic change to: {new_subject}")
                return False, 0.0, metadata

        # Fallback: pattern detected = continuation (without similarity validation)
        metadata['reason'] = 'pattern_match_only'
        return True, 0.7, metadata

    def batch_classify(self, queries: List[str]) -> List[QueryClassification]:
        """
        Classify multiple queries
        
        Args:
            queries: List of query texts
            
        Returns:
            List of QueryClassification objects
        """
        logger.info(f"Batch classifying {len(queries)} queries")
        return [self.classify(query) for query in queries]


# Convenience function
def create_query_classifier(**kwargs) -> QueryClassifier:
    """
    Convenience function to create a query classifier
    
    Args:
        **kwargs: Arguments for QueryClassifier
        
    Returns:
        QueryClassifier instance
    """
    return QueryClassifier(**kwargs)
