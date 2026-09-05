"""
User Query Handler - Step 1 of Query Processing
Handles query intake, validation, logging, and preprocessing
"""

import re
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetadata:
    """Metadata for tracking queries"""
    query_id: str
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_length: int = 0
    detected_language: str = "en"
    validation_status: str = "valid"
    validation_message: Optional[str] = None
    processing_stage: str = "received"
    # Chitchat detection fields
    is_chitchat: bool = False
    chitchat_type: Optional[str] = None
    canned_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class Query:
    """
    Structured query object with metadata
    """
    text: str
    metadata: QueryMetadata
    is_valid: bool = True
    preprocessing_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "is_valid": self.is_valid,
            "preprocessing_applied": self.preprocessing_applied
        }


class QueryValidator:
    """
    Validates incoming queries for format, length, and safety.
    Also detects greetings/chitchat to avoid expensive RAG pipeline.
    """

    # Greeting patterns (exact matches, case-insensitive)
    GREETINGS = {
        "hi", "hey", "hello", "yo", "sup", "greetings",
        "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "whats up", "how's it going",
        "hows it going", "howdy", "hiya"
    }

    # Acknowledgment patterns
    ACKNOWLEDGMENTS = {
        "thanks", "thank you", "ok", "okay", "got it",
        "yes", "no", "yeah", "yep", "nope", "sure",
        "alright", "cool", "great", "perfect", "awesome"
    }

    # Meaningless test queries
    MEANINGLESS = {
        "test", "testing", "hello world", "test test",
        ".", "..", "...", "?", "??"
    }

    # # Canned responses for each chitchat type
    # CANNED_RESPONSES = {
    #     "greeting": "Hello! I'm Synapse, developed by Eris AI. I specialize in California Family Law. How can I help you today?",
    #     "acknowledgment": "You're welcome! Is there anything else about California Family Law I can help you with?",
    #     "meaningless": "I'm here to help with California Family Law questions. What would you like to know about divorce, custody, support, or property division?"
    # }

    DEFAULT_CANNED_RESPONSES = {
        "greeting": "Hello! How can I help you today?",
        "acknowledgment": "You're welcome! Is there anything else I can help with?",
        "meaningless": "I'm here to help with questions related to this knowledge base.",
        "legal_determination": (
            "The documents do not make determinations of guilt or innocence. "
            "I can summarize what the documents state and cite the sources."
        ),
    }

    def __init__(
            self,
            min_length: int = 1,
            max_length: int = 4000,
            allow_empty: bool = False,
            check_profanity: bool = False,
            canned_responses: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize validator
        
        Args:
            min_length: Minimum query length in characters
            max_length: Maximum query length in characters
            allow_empty: Whether to allow empty queries
            check_profanity: Whether to check for inappropriate content
        """
        self.min_length = min_length
        self.max_length = max_length
        self.allow_empty = allow_empty
        self.check_profanity = check_profanity
        self.canned_responses = canned_responses or dict(self.DEFAULT_CANNED_RESPONSES)

        # Patterns for validation
        self.suspicious_patterns = [
            r'<script',  # XSS attempts
            r'javascript:',
            r'eval\(',
            r'exec\(',
        ]

        logger.info("QueryValidator initialized")
        logger.info(f"  - Min length: {min_length}")
        logger.info(f"  - Max length: {max_length}")
        logger.info(f"  - Greeting detection: enabled")

    def validate(self, query_text: str) -> tuple[bool, Optional[str]]:
        """
        Validate query text
        
        Args:
            query_text: The query text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if None
        if query_text is None:
            return False, "Query cannot be None"

        # Convert to string if needed
        query_text = str(query_text).strip()

        # Check empty
        if not query_text:
            if self.allow_empty:
                return True, None
            return False, "Query cannot be empty"

        # Check length
        if len(query_text) < self.min_length:
            return False, f"Query too short (min {self.min_length} characters)"

        if len(query_text) > self.max_length:
            return False, f"Query too long (max {self.max_length} characters)"

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                return False, "Query contains potentially unsafe content"

        # Check if only special characters
        if re.match(r'^[^\w\s]+$', query_text):
            return False, "Query contains only special characters"

        return True, None

    def detect_chitchat(self, query_text: str) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Enhanced chitchat detection with proper priority.
        
        This is a fast, rule-based check that happens before expensive retrieval.
        
        Priority order:
        1. Check for question indicators FIRST (if found, NOT chitchat)
        2. Check for exact chitchat matches (greetings, acknowledgments)
        3. Check for greeting phrases with friendly additions only
        
        Args:
            query_text: Query text to check
            
        Returns:
            Tuple of (is_chitchat, chitchat_type, canned_response)
            - is_chitchat: True if this is chitchat
            - chitchat_type: "greeting", "acknowledgment", or "meaningless"
            - canned_response: Pre-defined response to return
        """
        if not query_text:
            return False, None, None

        def _canned(key: str) -> Optional[str]:
            return self.canned_responses.get(key) or self.DEFAULT_CANNED_RESPONSES.get(key)

        # Normalize for matching
        query_lower = query_text.lower().strip()
        if not query_lower:
            return False, None, None

        # Fast-path direct meaningless checks (punctuation-only etc.)
        if query_lower in self.MEANINGLESS:
            logger.info(f"Detected meaningless query: '{query_text}'")
            return True, "meaningless", _canned("meaningless")

        # Tokenize robustly so punctuation variants still match:
        #   "hello," -> ["hello"], "hi,who" -> ["hi", "who"], "what's" preserved
        words = re.findall(r"[a-z0-9']+", query_lower)
        query_compact = " ".join(words).strip()
        query_compact_no_apostrophe = query_compact.replace("'", "")

        # Single-token, very short queries like "h" or "j" are meaningless
        if len(words) == 1:
            token = words[0]
            if len(token) <= 2:
                if token not in self.GREETINGS and token not in self.ACKNOWLEDGMENTS:
                    logger.info(f"Detected meaningless short query: '{query_text}'")
                    return True, "meaningless", _canned("meaningless")

        # PRIORITY 1: Check for question indicators FIRST
        # If query has question words, it's NOT chitchat regardless of greeting
        question_indicators = {
            'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose',
            'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does',
            'tell', 'show', 'explain', 'describe', 'find', 'search'
        }

        # Check for question words in query
        if any(word in question_indicators for word in words):
            logger.debug(f"Query contains question indicators, not chitchat: '{query_text}'")
            return False, None, None

        # Check for question phrases (e.g., "tell me")
        question_phrases = {'tell me', 'show me'}
        if any(phrase in query_compact for phrase in question_phrases):
            logger.debug(f"Query contains question phrases, not chitchat: '{query_text}'")
            return False, None, None

        # PRIORITY 2: Now check for pure chitchat
        normalized_variants = {query_compact, query_compact_no_apostrophe}

        # Exact matches in greetings
        if any(variant in self.GREETINGS for variant in normalized_variants):
            logger.info(f"Detected greeting: '{query_text}'")
            return True, "greeting", _canned("greeting")

        # Exact matches in acknowledgments
        if any(variant in self.ACKNOWLEDGMENTS for variant in normalized_variants):
            logger.info(f"Detected acknowledgment: '{query_text}'")
            return True, "acknowledgment", _canned("acknowledgment")

        # Exact matches in meaningless
        if any(variant in self.MEANINGLESS for variant in normalized_variants):
            logger.info(f"Detected meaningless query: '{query_text}'")
            return True, "meaningless", _canned("meaningless")

        # PRIORITY 3: Greeting phrases (hi there, hey again, etc.)
        if len(words) >= 2 and len(words) <= 3:
            # First word is greeting
            if words[0] in {"hi", "hey", "hello", "yo"}:
                # Check if rest is just friendly additions
                rest = words[1:]
                if all(word in {"there", "again", "buddy", "friend", "mate", "pal"} for word in rest):
                    logger.info(f"Detected greeting phrase: '{query_text}'")
                    return True, "greeting", _canned("greeting")

            # First two words are greeting
            if len(words) >= 2:
                first_two = " ".join(words[:2])
                if first_two in {"good morning", "good afternoon", "good evening"}:
                    # If nothing after or just friendly words
                    if len(words) == 2:
                        logger.info(f"Detected greeting phrase: '{query_text}'")
                        return True, "greeting", _canned("greeting")

        # Symbol-heavy noise (e.g., "#", "@#($*@#", "@S", "@#$SDF#@$") is meaningless
        if self._looks_like_symbol_noise(query_lower):
            logger.info(f"Detected meaningless symbol-noise query: '{query_text}'")
            return True, "meaningless", _canned("meaningless")

        # Not chitchat
        return False, None, None

    @staticmethod
    def _looks_like_symbol_noise(query_lower: str) -> bool:
        """
        Detect queries that are mostly symbols with minimal alphanumeric content.

        Examples: "#", "@#($*@#", "@S", "@#$SDF#@$"
        """
        if not query_lower:
            return False

        # Require at least one non-word symbol (punctuation, symbols, etc.)
        if not re.search(r"[^\w\s]", query_lower):
            return False

        # Count alphanumeric characters (letters/digits) only
        alnum_total = sum(len(run) for run in re.findall(r"[a-z0-9]+", query_lower))

        # If there are no alphanumerics, it's pure symbol noise
        if alnum_total == 0:
            return True

        # If alphanumerics are very short and surrounded by symbols, treat as noise
        return alnum_total <= 3

    def detect_legal_determination(self, query_text: str) -> tuple[bool, Optional[str]]:
        """
        Detect if query is asking about guilt, innocence, or legal determinations.

        These questions should receive a canned response explaining that documents
        don't make determinations of guilt/innocence.

        Args:
            query_text: Query text to check

        Returns:
            Tuple of (is_legal_determination, canned_response)
        """
        if not query_text:
            return False, None

        # Normalize for matching
        query_lower = query_text.lower().strip()

        # Keywords indicating legal determination questions
        legal_keywords = {
            "guilty", "guilt", "innocent", "innocence", "commit", "committed",
            "abuse", "abused", "crime", "crimes", "criminal", "knew what",
            "responsible", "liability", "liable", "culpable", "culpability"
        }

        # Question patterns about guilt/innocence
        determination_patterns = [
            r'\b(did|has|have)\s+\w+\s+(commit|committed)',  # "did X commit"
            r'\b(did|has|have)\s+\w+\s+abuse',  # "did X abuse"
            r'\bwho\s+(is|was|are|were)\s+(guilty|innocent)',  # "who is guilty/innocent"
            r'\b(guilty|innocent)\s+of',  # "guilty of", "innocent of"
            r'\bwho\s+knew\s+what',  # "who knew what"
            r'\bis\s+\w+\s+(guilty|innocent|responsible)',  # "is X guilty/innocent/responsible"
            r'\bwas\s+\w+\s+(guilty|innocent|responsible)',  # "was X guilty/innocent/responsible"
            r'\b(committed|commit)\s+(crimes|criminal)',  # "committed crimes"
        ]

        # Check if query contains legal keywords
        has_legal_keyword = any(keyword in query_lower for keyword in legal_keywords)

        if has_legal_keyword:
            # Check if it matches determination patterns
            for pattern in determination_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Detected legal determination query: '{query_text}'")
                    return True, (
                        self.canned_responses.get("legal_determination")
                        or self.DEFAULT_CANNED_RESPONSES.get("legal_determination")
                    )

        return False, None

    @staticmethod
    def detect_language(query_text: str) -> str:
        """
        Simple language detection (can be enhanced with langdetect library)

        Args:
            query_text: Query text

        Returns:
            Language code (default: 'en')
        """
        # Simple heuristic: check for non-ASCII characters
        # TODO: consider using langdetect library

        # Check if ASCII only
        if query_text.isascii():
            return "en"

        # Basic checks for common languages
        # Chinese/Japanese/Korean
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', query_text):
            return "cjk"

        # Arabic
        if re.search(r'[\u0600-\u06ff]', query_text):
            return "ar"

        # Cyrillic (Russian, etc)
        if re.search(r'[\u0400-\u04ff]', query_text):
            return "ru"

        # Default to English
        return "en"

    def strip_greeting_words(self, query_text: str) -> str:
        """
        Strip greeting words from the beginning of a query for better retrieval.
        
        This is used AFTER chitchat detection to clean queries that contain
        both greetings and real questions (e.g., "hi who is Jeremy Salvador?" → "who is Jeremy Salvador?")
        
        Args:
            query_text: Query text that may start with greetings
            
        Returns:
            Query with greeting words removed from the start
        """
        if not query_text:
            return query_text

        # Split into words (preserve original case for reconstruction)
        words = query_text.split()

        if not words:
            return query_text

        # Greeting words to strip from the beginning
        greeting_starters = {"hi", "hey", "hello", "yo", "sup", "howdy", "hiya"}
        multi_word_greetings = {"good morning", "good afternoon", "good evening"}

        # Track how many words to skip
        skip_count = 0

        # Check for multi-word greetings first (handle punctuation)
        if len(words) >= 2:
            # Remove punctuation from words for comparison
            first_word_clean = words[0].lower().rstrip(',.!?;:')
            second_word_clean = words[1].lower().rstrip(',.!?;:')
            first_two_clean = f"{first_word_clean} {second_word_clean}"

            if first_two_clean in multi_word_greetings:
                skip_count = 2

        # Check for single-word greetings (handle punctuation)
        if skip_count == 0:
            first_word_clean = words[0].lower().rstrip(',.!?;:')
            if first_word_clean in greeting_starters:
                skip_count = 1

        # Prepare the query for filler stripping
        if skip_count > 0:
            # Get the remaining words after greeting
            remaining_words = words[skip_count:]

            if not remaining_words:
                # Don't strip if it would leave empty query
                return query_text

            # Join back and clean up
            cleaned = " ".join(remaining_words).strip()

            # Remove leading comma or punctuation if present
            cleaned = cleaned.lstrip(',.;:!?').strip()
        else:
            # No greeting to strip, use original query
            cleaned = query_text.strip()

        # Strip common conversational filler phrases that reduce retrieval quality
        # This happens whether or not we stripped a greeting
        # e.g., "can you tell me who is X?" → "who is X?"
        conversational_fillers = [
            "can you tell me",
            "could you tell me",
            "can you explain",
            "could you explain",
            "please tell me",
            "please explain",
            "would you tell me",
            "would you explain",
            "can you please tell me",
            "could you please tell me",
            "can you please explain",
            "could you please explain",
        ]

        cleaned_lower = cleaned.lower()
        for filler in conversational_fillers:
            if cleaned_lower.startswith(filler):
                # Strip the filler phrase
                cleaned = cleaned[len(filler):].strip()
                # Remove any leading punctuation again
                cleaned = cleaned.lstrip(',.;:!?').strip()
                logger.debug(f"Stripped conversational filler: '{filler}'")
                break

        # Only return cleaned version if something was actually stripped
        if cleaned != query_text:
            logger.debug(f"Stripped from query: '{query_text}' → '{cleaned}'")
            return cleaned if cleaned else query_text

        return query_text


class QueryLogger:
    """
    Logs queries with metadata for tracking and analysis
    """

    def __init__(
            self,
            log_dir: str = "./logs/queries",
            log_to_file: bool = True,
            log_to_console: bool = True
    ):
        """
        Initialize query logger
        
        Args:
            log_dir: Directory to store query logs
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console

        if self.log_to_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.query_log_file = self.log_dir / "queries.jsonl"
            logger.info(f"Query logs will be saved to: {self.query_log_file}")

    def log_query(self, query: Query) -> None:
        """
        Log query with metadata
        
        Args:
            query: Query object to log
        """
        log_entry = {
            "query_id": query.metadata.query_id,
            "timestamp": query.metadata.timestamp,
            "query_text": query.text,
            "query_length": query.metadata.query_length,
            "user_id": query.metadata.user_id,
            "session_id": query.metadata.session_id,
            "detected_language": query.metadata.detected_language,
            "is_valid": query.is_valid,
            "validation_status": query.metadata.validation_status,
            "validation_message": query.metadata.validation_message,
            "preprocessing_applied": query.preprocessing_applied
        }

        # Log to console
        if self.log_to_console:
            logger.info(
                f"Query logged: {query.metadata.query_id} | Length: {query.metadata.query_length} | Valid: {query.is_valid}")

        # Log to file
        if self.log_to_file:
            try:
                with open(self.query_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.error(f"Failed to write query log: {e}")


class QueryPreprocessor:
    """
    Preprocesses queries (normalization, cleaning)
    """

    def __init__(
            self,
            normalize_whitespace: bool = True,
            remove_special_chars: bool = False,
            lowercase: bool = False
    ):
        """
        Initialize preprocessor
        
        Args:
            normalize_whitespace: Normalize whitespace
            remove_special_chars: Remove special characters
            lowercase: Convert to lowercase
        """
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase

    def preprocess(self, query_text: str) -> tuple[str, List[str]]:
        """
        Preprocess query text
        
        Args:
            query_text: Raw query text
            
        Returns:
            Tuple of (preprocessed_text, list of applied operations)
        """
        processed = query_text
        operations = []

        # Normalize whitespace
        if self.normalize_whitespace:
            processed = ' '.join(processed.split())
            operations.append("normalize_whitespace")

        # Remove excessive special characters
        if self.remove_special_chars:
            # Keep basic punctuation but remove excessive special chars
            processed = re.sub(r'[^\w\s?.,!\'"-]', ' ', processed)
            processed = ' '.join(processed.split())
            operations.append("remove_special_chars")

        # Lowercase (usually not recommended for queries)
        if self.lowercase:
            processed = processed.lower()
            operations.append("lowercase")

        return processed, operations


class QueryHandler:
    """
    Main query handler that orchestrates validation, logging, and preprocessing
    """

    def __init__(
            self,
            min_query_length: int = 1,
            max_query_length: int = 4000,
            log_queries: bool = True,
            log_dir: str = "./logs/queries",
            preprocess_queries: bool = True,
            enable_rate_limiting: bool = False,
            rate_limit_per_user: int = 100,  # requests per minute
            canned_responses: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize query handler
        
        Args:
            min_query_length: Minimum query length
            max_query_length: Maximum query length
            log_queries: Whether to log queries
            log_dir: Directory for query logs
            preprocess_queries: Whether to preprocess queries
            enable_rate_limiting: Enable rate limiting (requires Redis for production)
            rate_limit_per_user: Max requests per user per minute
        """
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.log_queries = log_queries
        self.preprocess_queries = preprocess_queries
        self.enable_rate_limiting = enable_rate_limiting
        self.rate_limit_per_user = rate_limit_per_user

        # Initialize components
        self.validator = QueryValidator(
            min_length=min_query_length,
            max_length=max_query_length,
            canned_responses=canned_responses,
        )

        if log_queries:
            self.logger = QueryLogger(log_dir=log_dir)

        if preprocess_queries:
            self.preprocessor = QueryPreprocessor(
                normalize_whitespace=True,
                remove_special_chars=False,
                lowercase=False
            )

        # Simple in-memory rate limiting (use Redis for production)
        self.request_counts: Dict[str, List[float]] = {}

        logger.info("=" * 80)
        logger.info("QueryHandler initialized")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  - Min query length: {min_query_length}")
        logger.info(f"  - Max query length: {max_query_length}")
        logger.info(f"  - Query logging: {log_queries}")
        logger.info(f"  - Query preprocessing: {preprocess_queries}")
        logger.info(f"  - Rate limiting: {enable_rate_limiting}")
        logger.info("=" * 80)

    @staticmethod
    def generate_query_id(query_text: str, timestamp: str) -> str:
        """
        Generate unique query ID
        
        Args:
            query_text: Query text
            timestamp: Timestamp string
            
        Returns:
            Unique query ID
        """
        content = f"{query_text}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def check_rate_limit(self, user_id: Optional[str]) -> tuple[bool, Optional[str]]:
        """
        Check if user has exceeded rate limit
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not self.enable_rate_limiting:
            return True, None

        if user_id is None:
            user_id = "anonymous"

        current_time = time.time()
        window_start = current_time - 60  # 1-minute window

        # Clean old requests
        if user_id in self.request_counts:
            self.request_counts[user_id] = [
                req_time for req_time in self.request_counts[user_id]
                if req_time > window_start
            ]
        else:
            self.request_counts[user_id] = []

        # Check limit
        if len(self.request_counts[user_id]) >= self.rate_limit_per_user:
            return False, f"Rate limit exceeded ({self.rate_limit_per_user} requests per minute)"

        # Add current request
        self.request_counts[user_id].append(current_time)

        return True, None

    def handle_query(
            self,
            query_text: str,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None
    ) -> Query:
        """
        Main method to handle incoming query
        
        Args:
            query_text: Raw query text from user
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            Query object with validation status and metadata
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        query_id = self.generate_query_id(query_text, timestamp)

        logger.info(f"Handling query: {query_id}")

        # Check rate limit
        rate_limit_ok, rate_limit_msg = self.check_rate_limit(user_id)
        if not rate_limit_ok:
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            metadata = QueryMetadata(
                query_id=query_id,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                query_length=len(query_text) if query_text else 0,
                validation_status="rate_limit_exceeded",
                validation_message=rate_limit_msg
            )
            query = Query(
                text=query_text,
                metadata=metadata,
                is_valid=False
            )
            if self.log_queries:
                self.logger.log_query(query)
            return query

        # Check for chitchat BEFORE validation (fast path)
        is_chitchat, chitchat_type, canned_response = self.validator.detect_chitchat(query_text)

        if is_chitchat:
            logger.info(f"Chitchat detected ({chitchat_type}), returning canned response")
            metadata = QueryMetadata(
                query_id=query_id,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                query_length=len(query_text) if query_text else 0,
                validation_status="chitchat",
                validation_message=f"Detected as {chitchat_type}",
                is_chitchat=True,
                chitchat_type=chitchat_type,
                canned_response=canned_response
            )
            query = Query(
                text=query_text,
                metadata=metadata,
                is_valid=True  # Chitchat is "valid" but doesn't need RAG pipeline
            )
            if self.log_queries:
                self.logger.log_query(query)
            return query

        # Check for legal determination questions BEFORE validation (fast path) (Epstein)
        is_legal_determination, legal_response = self.validator.detect_legal_determination(query_text)

        if is_legal_determination:
            logger.info(f"Legal determination query detected, returning canned response")
            metadata = QueryMetadata(
                query_id=query_id,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                query_length=len(query_text) if query_text else 0,
                validation_status="legal_determination",
                validation_message="Query asks about guilt/innocence determination",
                is_chitchat=True,  # Treat as chitchat for pipeline handling
                chitchat_type="legal_determination",
                canned_response=legal_response
            )
            query = Query(
                text=query_text,
                metadata=metadata,
                is_valid=True  # Valid but doesn't need RAG pipeline
            )
            if self.log_queries:
                self.logger.log_query(query)
            return query

        # Validate query
        is_valid, validation_message = self.validator.validate(query_text)

        if not is_valid:
            logger.warning(f"Query validation failed: {validation_message}")
            metadata = QueryMetadata(
                query_id=query_id,
                timestamp=timestamp,
                user_id=user_id,
                session_id=session_id,
                query_length=len(query_text) if query_text else 0,
                validation_status="invalid",
                validation_message=validation_message
            )
            query = Query(
                text=query_text,
                metadata=metadata,
                is_valid=False
            )
            if self.log_queries:
                self.logger.log_query(query)
            return query

        # Preprocess query
        processed_text = query_text.strip()
        preprocessing_applied = []

        if self.preprocess_queries:
            processed_text, preprocessing_applied = self.preprocessor.preprocess(query_text)
            logger.debug(f"Preprocessing applied: {preprocessing_applied}")

        # Detect language
        detected_language = self.validator.detect_language(processed_text)

        # Create metadata
        metadata = QueryMetadata(
            query_id=query_id,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            query_length=len(processed_text),
            detected_language=detected_language,
            validation_status="valid",
            validation_message=None,
            processing_stage="validated"
        )

        # Create query object
        query = Query(
            text=processed_text,
            metadata=metadata,
            is_valid=True,
            preprocessing_applied=preprocessing_applied
        )

        # Log query
        if self.log_queries:
            self.logger.log_query(query)

        logger.info(f" Query handled successfully: {query_id}")
        logger.info(f"   Length: {metadata.query_length} | Language: {detected_language}")

        return query

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get handler statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.log_queries and hasattr(self, 'logger'):
            try:
                with open(self.logger.query_log_file, 'r') as f:
                    total_queries = sum(1 for _ in f)
            except FileNotFoundError:
                total_queries = 0
        else:
            total_queries = 0

        return {
            "total_queries_logged": total_queries,
            "rate_limiting_enabled": self.enable_rate_limiting,
            "active_users": len(self.request_counts) if self.enable_rate_limiting else 0
        }


# Convenience function
def create_query_handler(**kwargs) -> QueryHandler:
    """
    Convenience function to create a query handler
    
    Args:
        **kwargs: Arguments for QueryHandler
        
    Returns:
        QueryHandler instance
    """
    return QueryHandler(**kwargs)
