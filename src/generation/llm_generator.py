"""
LLM Generator for RAG Pipeline.

This module provides LLM integration for answer generation with Haystack-style interface.

Features:
- Local model inference with vllm
- Configurable generation parameters
- Stop sequences support
- Error handling and timeout protection
# vLLM handles stop sequences natively so we don't need StopOnTokens class
"""

import re
import os
import time
import threading
import torch
import logging
from dataclasses import dataclass
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("vllm not available. Install with: pip install vllm")

GEMINI_AVAILABLE = False


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 3072  # Increased from 2048 for more comprehensive responses
    temperature: float = 0.3  # Lower for factual accuracy
    top_p: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.2  # Increased to prevent repetitions
    device: str = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
    timeout: int = 60  # seconds
    normalize_newlines: str = "preserve"  # Options: "preserve", "single", "remove"
    # preserve: Keep \n\n for paragraphs (default)
    # single: Convert all \n\n to \n (no paragraph breaks)
    # remove: Replace all \n with spaces (continuous text)

    frequency_penalty: float = 0.3  # Penalize token frequency
    presence_penalty: float = 1.0  # Penalize token presence
    # LLM admission control (Phase 0.6)
    max_concurrency: int = int(os.environ.get("LLM_MAX_CONCURRENCY", "2"))
    admission_timeout_seconds: float = float(os.environ.get("LLM_ADMISSION_TIMEOUT_SECONDS", "6.0"))
    retry_after_seconds: int = int(os.environ.get("LLM_RETRY_AFTER_SECONDS", "2"))

    # Backend and model configurations for API integrations
    llm_backend: str = "local"  # options: "local" or "api"
    api_model_name: str = "gemini-2.5-flash-lite"


class LLMOverloadedError(RuntimeError):
    """Raised when LLM generation capacity is saturated."""

    def __init__(
            self,
            message: str,
            retry_after_seconds: int = 2,
            *,
            inflight: Optional[int] = None,
            max_concurrency: Optional[int] = None,
            waiters: Optional[int] = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.inflight = inflight
        self.max_concurrency = max_concurrency
        self.waiters = waiters


class LLMGenerator:
    """
    LLM Generator for RAG answer generation.

    Uses an LLM, accurate generation with citations.
    Haystack-compatible interface for easy integration.
    """

    def __init__(
            self,
            config: Optional[GenerationConfig] = None,
            existing_model=None,
            existing_tokenizer=None
    ):
        """
        Initialize LLM Generator.

        Args:
            config: Generation configuration
            existing_model: Reuse existing model (saves memory)
            existing_tokenizer: Reuse existing tokenizer (not used in vLLM)
        """
        self.config = config or GenerationConfig()
        self.config.max_concurrency = max(1, int(self.config.max_concurrency))
        self.config.admission_timeout_seconds = max(0.05, float(self.config.admission_timeout_seconds))
        self.config.retry_after_seconds = max(1, int(self.config.retry_after_seconds))

        # Forcing local backend as API/Gemini integration has been removed
        configured_backend = os.environ.get("LLM_BACKEND", self.config.llm_backend).lower()
        if configured_backend == "api":
            logger.warning("Gemini API backend is removed. Falling back to 'local' backend.")
        self.backend = "local"
        self.api_model_name = ""

        logger.info("   Initializing LLMGenerator")
        logger.info(f"   Backend: {self.backend}")
        logger.info(f"   Model: {self.config.model_name}")
        logger.info(f"   Device: {self.config.device}")
            
        logger.info(f"   Max tokens: {self.config.max_new_tokens}")
        logger.info(f"   Base/default temperature: {self.config.temperature}")
        logger.info(
            "   LLM admission control: max_concurrency=%s, acquire_timeout=%.2fs",
            self.config.max_concurrency,
            self.config.admission_timeout_seconds,
        )

        # Guard vLLM access under controlled concurrency to avoid GPU contention spikes.
        self._generation_semaphore = threading.BoundedSemaphore(self.config.max_concurrency)
        self._state_lock = threading.Lock()
        # vLLM's Python LLM facade is not safe for overlapping generate() calls from
        # multiple request threads on the same instance. We still admit multiple
        # requests concurrently so retrieval/routing can overlap, but actual model
        # access is serialized to prevent cross-request prompt/response bleed.
        self._model_access_lock = threading.Lock()
        self._inflight = 0
        self._waiters = 0
        self._rejections = 0
        self._peak_inflight = 0
        self._inflight_started_at: Dict[int, float] = {}
        self._inflight_ticket_seq = 0

        self.api_client = None

        # Reuse existing model if provided, or load locally if local backend selected
        self.model = None
        if self.backend == "local":
            if existing_model:
                logger.info(" Reusing existing model")
                self.model = existing_model
            else:
                self._load_model()

        logger.info(" LLMGenerator initialized successfully")

    def _load_model(self):
        """Load the LLM model with vLLM."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "vLLM is required for local LLM generation. "
                "Install with: pip install vllm"
            )
        logger.info(f" Loading model: {self.config.model_name}")

        quantization_type = None
        if "AWQ" in self.config.model_name.upper():
            quantization_type = "awq"
        elif "FP8" in self.config.model_name.upper():
            quantization_type = "fp8"

        try:
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,  # Reduced from 0.8 to provide ~4GB headroom
                max_model_len=8192,  # Reduced from 32768 to save memory
                trust_remote_code=True,
                quantization=quantization_type,
                dtype="auto",
                enable_prefix_caching=True,
                swap_space=4,  # Add 4GB CPU swap space for overflow handling
            )
            logger.info(f" Model loaded successfully")
        except Exception as e:
            logger.error(f" Failed to load model: {e}")
            raise

    def get_concurrency_metrics(self) -> Dict[str, Any]:
        """Expose LLM admission/concurrency metrics for readiness and diagnostics."""
        with self._state_lock:
            inflight = self._inflight
            waiters = self._waiters
            rejections = self._rejections
            peak_inflight = self._peak_inflight
            now = time.monotonic()
            oldest_inflight_seconds = (
                max(0.0, now - min(self._inflight_started_at.values()))
                if self._inflight_started_at
                else 0.0
            )

        saturation = inflight / self.config.max_concurrency if self.config.max_concurrency > 0 else 1.0
        return {
            "enabled": True,
            "max_concurrency": self.config.max_concurrency,
            "model_access_serialized": True,
            "inflight": inflight,
            "waiters": waiters,
            "rejections": rejections,
            "peak_inflight": peak_inflight,
            "admission_timeout_seconds": self.config.admission_timeout_seconds,
            "retry_after_seconds": self.config.retry_after_seconds,
            "saturation": round(min(max(saturation, 0.0), 1.0), 3),
            "oldest_inflight_seconds": round(oldest_inflight_seconds, 3),
        }

    @contextmanager
    def _generation_slot(self):
        """Acquire bounded LLM generation capacity or fail fast with overload signal."""
        acquired = False
        inflight_ticket: Optional[int] = None
        with self._state_lock:
            self._waiters += 1

        try:
            acquired = self._generation_semaphore.acquire(timeout=self.config.admission_timeout_seconds)
            if not acquired:
                with self._state_lock:
                    self._rejections += 1
                    inflight_snapshot = self._inflight
                    waiters_snapshot = max(0, self._waiters - 1)
                raise LLMOverloadedError(
                    (
                        "LLM generation queue is saturated "
                        f"(inflight={inflight_snapshot}, max={self.config.max_concurrency})"
                    ),
                    retry_after_seconds=self.config.retry_after_seconds,
                    inflight=inflight_snapshot,
                    max_concurrency=self.config.max_concurrency,
                    waiters=waiters_snapshot,
                )
        finally:
            with self._state_lock:
                self._waiters = max(0, self._waiters - 1)

        with self._state_lock:
            self._inflight += 1
            if self._inflight > self._peak_inflight:
                self._peak_inflight = self._inflight
            self._inflight_ticket_seq += 1
            inflight_ticket = self._inflight_ticket_seq
            self._inflight_started_at[inflight_ticket] = time.monotonic()

        try:
            yield
        finally:
            with self._state_lock:
                self._inflight = max(0, self._inflight - 1)
                if inflight_ticket is not None:
                    self._inflight_started_at.pop(inflight_ticket, None)
            if acquired:
                self._generation_semaphore.release()

    def generate(
            self,
            prompt: str,
            max_new_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
            stop_sequences: Optional[List[str]] = None,
            **kwargs
    ) -> str:
        """
        Generate answer from prompt.

        Args:
            prompt: Input prompt (from RAGPromptBuilder)
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            stop_sequences: List of stop sequences (e.g., ["Question:", "Context:"])
            **kwargs: Additional generation parameters

        Returns:
            Generated answer text
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        start_time = time.time()



        # Route to local vLLM backend
        try:
            purpose = kwargs.get("purpose")
            purpose_tag = f"[{purpose}] " if purpose else ""
            with self._generation_slot():
                logger.info(f" {purpose_tag}Generating answer locally (prompt length: {len(prompt)} chars)")

                # Lazy load local model on demand
                if self.model is None:
                    logger.info("   Lazy loading local vLLM model...")
                    self._load_model()

                # Determine min_tokens based on semantic query complexity
                # Use the requires_long_response flag from query classification
                if "min_tokens" in kwargs:
                    raw_min_tokens = kwargs.get("min_tokens")
                    min_tokens = int(raw_min_tokens) if raw_min_tokens else 0
                elif kwargs.get("requires_long_response", False):
                    # Default long-response floor when enabled
                    min_tokens = 512
                    logger.info("   Complex query detected (semantic analysis) - using min_tokens=512")
                else:
                    min_tokens = 0  # Let model decide for simple queries

                # vLLM sampling parameters
                long_response_max_tokens = kwargs.get("long_response_max_tokens")
                if min_tokens > 0 and long_response_max_tokens:
                    resolved_max_tokens = int(long_response_max_tokens)
                elif min_tokens > 0:
                    resolved_max_tokens = 2048
                else:
                    resolved_max_tokens = max_new_tokens or self.config.max_new_tokens

                with self._model_access_lock:
                    logger.info("   %sWaiting for exclusive model access", purpose_tag)
                    tokenizer = self.model.get_tokenizer()
                    sampling_params = SamplingParams(
                        temperature=temperature or self.config.temperature,
                        top_p=kwargs.get("top_p", self.config.top_p),
                        max_tokens=resolved_max_tokens,
                        min_tokens=min_tokens,
                        repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                        # Explicitly prevent code-mode activation
                        stop=(stop_sequences or []) + [
                            "\n```",
                            "\n\n```",
                            "```python",
                            "```json",
                            "```markdown",
                        ],
                        stop_token_ids=[
                            tokenizer.eos_token_id,
                            tokenizer.encode("<|im_end|>")[0],
                        ],
                        skip_special_tokens=True,
                        include_stop_str_in_output=False,
                        presence_penalty=self.config.presence_penalty,
                        frequency_penalty=self.config.frequency_penalty,
                    )

                    logger.info(
                        f"   {purpose_tag}SamplingParams: max_tokens={sampling_params.max_tokens}, "
                        f"min_tokens={min_tokens}, temp={sampling_params.temperature}"
                    )

                    # Use vLLM's generate() for all models. Shared-model access is serialized
                    # here because the Python facade can misroute overlapping threaded calls.
                    outputs = self.model.generate([prompt], sampling_params)
                raw_text = outputs[0].outputs[0].text

                # Log raw output for debugging
                logger.info(f"   Raw output length: {len(raw_text)} chars")

                # Clean up the output
                generated_text = self._post_process(raw_text, stop_sequences)

                elapsed_time = time.time() - start_time
                output_tokens = len(generated_text.split()) * 1.3  # Rough estimate

                logger.info(f"   Generation complete")
                logger.info(f"   Time: {elapsed_time:.2f}s")
                logger.info(f"   Output tokens: ~{int(output_tokens)}")
                logger.info(f"   Tokens/sec: {output_tokens / elapsed_time:.1f}")

                return generated_text

        except LLMOverloadedError:
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _aggressive_paragraph_stop(self, text: str) -> str:
        """
        Stop at FIRST paragraph boundary after meta-commentary detected.
        This prevents the double-answer problem.
        
        Args:
            text: Generated text
            
        Returns:
            Text stopped at appropriate paragraph boundary
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if not paragraphs:
            return text

        # Meta-commentary triggers (FIRST occurrence = stop)
        meta_triggers = {
            # Self-correction phrases
            "do not repeat": 0,
            "okay, let me": 0,
            "let me address": 0,
            "to clarify": 0,
            "in other words": 2,  # Allow 2 uses
            "put differently": 2,

            # Internal reasoning
            "wait,": 0,
            "actually,": 1,  # Allow 1 use
            "however, after": 0,
            "upon review": 0,
            "correction:": 0,

            # Instructions to self
            "keep response": 0,
            "make sure": 1,
            "ensure that": 1,
        }

        clean_paragraphs = []
        trigger_counts = {k: 0 for k in meta_triggers.keys()}

        for i, para in enumerate(paragraphs):
            para_lower = para.lower()

            # Check for meta-triggers
            stop_here = False
            for trigger, allowed_count in meta_triggers.items():
                if trigger in para_lower:
                    trigger_counts[trigger] += 1
                    if trigger_counts[trigger] > allowed_count:
                        logger.warning(f"Meta-commentary detected at paragraph {i}: '{trigger}'")
                        stop_here = True
                        break

            if stop_here:
                # Stop BEFORE this paragraph
                break

            clean_paragraphs.append(para)

            # Safety: If we have 5+ substantial paragraphs, that's probably enough
            if len(clean_paragraphs) >= 5 and all(len(p) > 100 for p in clean_paragraphs):
                logger.info(f"Natural stop at {len(clean_paragraphs)} paragraphs")
                break

        # Edge case: If we filtered everything, return first paragraph
        if not clean_paragraphs and paragraphs:
            return paragraphs[0]

        return '\n\n'.join(clean_paragraphs)

    def _detect_duplicate_response(self, text: str) -> tuple:
        """
        Detect if model gave answer twice (double-answer problem).
        
        Args:
            text: Generated text
            
        Returns:
            Tuple of (is_duplicate: bool, clean_text: str)
        """
        from difflib import SequenceMatcher

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p) > 50]

        if len(paragraphs) < 3:
            return False, text

        # Check for structural duplication
        # Common pattern: Intro -> Body -> "Let me clarify" -> REPEAT of body

        def similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        # Check if latter half is similar to first half
        mid_point = len(paragraphs) // 2
        first_half = ' '.join(paragraphs[:mid_point])
        second_half = ' '.join(paragraphs[mid_point:])

        if similarity(first_half, second_half) > 0.6:  # 60% similar
            logger.warning(f"Duplicate structure detected (similarity: {similarity(first_half, second_half):.2f})")
            # Return first half only
            return True, '\n\n'.join(paragraphs[:mid_point])

        # Method 2: Repeated key phrases
        # Count occurrences of distinctive phrases (4+ words)
        words_first = first_half.lower().split()
        words_second = second_half.lower().split()

        # Extract 4-grams
        def get_ngrams(words, n=4):
            return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

        ngrams_first = set(get_ngrams(words_first))
        ngrams_second = set(get_ngrams(words_second))

        overlap = ngrams_first & ngrams_second
        if len(overlap) > 10:  # More than 10 repeated 4-word phrases
            logger.warning(f"High n-gram overlap detected: {len(overlap)} repeated phrases")
            return True, '\n\n'.join(paragraphs[:mid_point])

        return False, text

    def _remove_code_artifacts(self, text: str) -> str:
        """
        Remove Markdown code blocks that appear mid-response.
        This handles the Qwen3-Coder contamination issue.
        
        Args:
            text: Generated text
            
        Returns:
            Text with code artifacts removed
        """
        # Pattern 1: Remove code blocks that appear AFTER substantial text
        # (Assumption: If there's 200+ chars before ```, it's likely an artifact)
        if len(text) > 200 and '```' in text:
            # Find first occurrence of ```
            code_block_start = text.find('```')

            # Check if there's substantial content before it
            before_code = text[:code_block_start].strip()

            if len(before_code) > 200:  # Substantial answer exists
                # Truncate at the code block
                logger.warning(f"Code block artifact detected at position {code_block_start}")

                # Find last complete sentence before code block
                last_period = max(
                    before_code.rfind('.'),
                    before_code.rfind('!'),
                    before_code.rfind('?')
                )

                if last_period > len(before_code) * 0.5:
                    return before_code[:last_period + 1]
                else:
                    return before_code

        # Pattern 2: Remove inline code formatting artifacts
        # Example: "text `code` more text" where code isn't meant to be code
        # Only if there are single backticks (not triple)
        if '`' in text and '```' not in text:
            # Remove single backticks (rare in legal text)
            text = text.replace('`', '')

        # Pattern 3: Remove "How does X..." follow-up questions
        # These appear when model gets confused and asks itself a new question
        follow_up_pattern = r'\n(?:How|What|Why|When|Where|Who) (?:does|is|are|was|were|do|did|can|could|should|would) .+?\?'

        matches = list(re.finditer(follow_up_pattern, text, re.IGNORECASE))
        if matches:
            # If follow-up question appears after substantial text, truncate
            first_match = matches[0]
            if first_match.start() > 200:  # Substantial content exists
                logger.warning(f"Self-generated follow-up question detected: {first_match.group()}")
                text = text[:first_match.start()].strip()

                # Ensure ends with punctuation
                if text and text[-1] not in '.!?':
                    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                    if last_period > len(text) * 0.5:
                        text = text[:last_period + 1]

        return text

    def _post_process(
            self,
            text: str,
            stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Post-process generated text to remove artifacts and repetitions.

        Args:
            text: Generated text
            stop_sequences: Stop sequences to remove

        Returns:
            Cleaned text
        """
        if not text:
            return text

        # PHASE 0: Aggressive paragraph-level early stopping (NEW - highest priority)
        text = self._aggressive_paragraph_stop(text)

        # PHASE 0.5: Detect and remove duplicate responses
        is_duplicate, text = self._detect_duplicate_response(text)
        if is_duplicate:
            logger.warning("Duplicate response structure removed")

        # PHASE 1: Remove code artifacts
        text = self._remove_code_artifacts(text)

        # PHASE 2: Fix escaped newlines and markdown formatting
        text = self._fix_markdown_formatting(text)

        # Replace em dashes with hyphens or commas
        text = re.sub(r'\s*—\s*', ', ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove emojis - the model sometimes adds these, especially when forced to generate more tokens
        # This regex covers most common emoji Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # misc symbols
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)

        # Clean up any leftover whitespace from emoji removal
        text = re.sub(r'  +', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n +', '\n', text)  # Space after newline
        text = re.sub(r' +\n', '\n', text)  # Space before newline

        # Handle multiple "Answer:" prefixes - take only the first answer
        if text.count('Answer:') > 1:
            parts = text.split('Answer:', 2)
            if len(parts) > 2:
                text = 'Answer:' + parts[1]

        # the stop sequence removal is handled by vLLM's SamplingParams.stop
        # we should not split on stop_sequences here as it causes false truncation
        # of valid content (e.g., "---" markdown separators)

        # AGGRESSIVE stop sequence and meta-commentary removal
        # vLLM's SamplingParams.stop should handle these, but sometimes the model
        # generates internal reasoning/verification after stop sequences
        # We need to explicitly truncate at these points

        # Truncate at explicit stop sequences and meta-commentary markers
        artifact_patterns = [
            '[End response]',  # Stop sequence that sometimes gets through
            '[End response.]',
            '[End of message]',  # Another stop sequence
            '\nEnd of message.',
            # Internal reasoning markers (LLM thinking out loud)
            '\nEnd message',
            'End message.',
            '\nWait -',
            '\nWait –',
            ' Wait -',
            'correction needed',
            'Correction needed',
            'Final Corrected Output',
            'Final Output:',
            'Without Any Violations',
            'per protocol',
            'This concludes',
            'No additional input required',
            'let me know how else',
            'within the defined parameters',
            # System prompt discussion
            ' The system prompt',
            '\nThe system prompt',
            ' All identifiers',
            '\nAll identifiers',
            # Compliance checking
            ' No violations',
            '\nNo violations',
            ' Response delivered',
            '\nResponse delivered',
            ' Final output',
            '\nFinal output',
            ' Proceeding with',
            '\nProceeding with',
            ' Confirmed operational',
            '\nConfirmed operational',
            ' System integrity',
            '\nSystem integrity',
            ' Waiting for next',
            '\nWaiting for next',
            # Note patterns
            '\n\nNote:',
            '\nNote:',
            '[N] notation',
            'The correct response should follow',
            'notation applies per source',
            'Answer: <answer here>',
            '\n\nAnswer:',  # Second answer block
            # NEW: Code/markdown artifact stops
            '\nOkay,',  # Self-correction pattern
            '\n\nOkay,',
        ]

        for pattern in artifact_patterns:
            if pattern in text:
                text = text.split(pattern)[0].strip()
                break  # Stop after first match to avoid over-truncation

        # Remove trailing [N] [N] [N] artifacts
        while text.endswith('[N]'):
            text = text[:-3].strip()

        # Clean up excessive [N] sequences (more than 3 in a row)
        text = re.sub(r'(\[N\]\s*){3,}', '', text)

        # Remove any repeated newlines (triple or more -> double)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        # Normalize paragraph spacing - ensure consistent double newlines for paragraphs
        # This prevents cases where LLM generates excessive spacing
        text = re.sub(r'\n{4,}', '\n\n', text)  # 4+ newlines -> 2 newlines

        # Clean up spacing around citations
        text = re.sub(r'\s+\[', ' [', text)  # Space before [
        text = re.sub(r'\]\s+', '] ', text)  # Space after ]

        # Remove incomplete sentences at the end
        if text and not text[-1] in '.!?':
            # Find last complete sentence
            last_punct = max(
                text.rfind('.'),
                text.rfind('!'),
                text.rfind('?')
            )
            if last_punct > len(text) * 0.5:  # Only if we have substantial content
                text = text[:last_punct + 1]

        # Apply newline normalization based on config
        normalize_mode = getattr(self.config, 'normalize_newlines', 'preserve')

        if normalize_mode == "single":
            # Convert all double newlines to single newlines (remove paragraph breaks)
            text = text.replace("\n\n", "\n")
        elif normalize_mode == "remove":
            # Replace all newlines with spaces (continuous text)
            text = text.replace("\n", " ")
            # Clean up multiple spaces
            text = re.sub(r'\s+', ' ', text)

        # # Fix markdown spacing for proper frontend rendering
        # # Ensure newline before horizontal rules (---)
        # text = re.sub(r'([^\n])\s*---', r'\1\n---', text)
        #
        # # Ensure newline before block quotes (>)
        # text = re.sub(r'([^\n])\s*>', r'\1\n>', text)

        # Detect and truncate degenerate text (semantic drift/rambling)
        text = self._truncate_degenerate_text(text)

        return text.strip()

    def _fix_markdown_formatting(self, text: str) -> str:
        """
        Fix common Markdown formatting issues from LLM output.

        This function uses a robust, pattern-based approach to detect and fix
        formatting issues without hard-coded rules for specific cases.

        Handles:
        1. Literal backslash-n sequences (\\n) appearing in text
        2. Missing blank lines before markdown special elements
        3. Improper spacing around punctuation
        4. List indentation issues

        Args:
            text: Generated text with potential formatting issues

        Returns:
            Text with corrected markdown formatting
        """
        import logging
        logger = logging.getLogger(__name__)

        if not text or len(text.strip()) == 0:
            return text

        # Debug: Log function entry
        # logger.info(f"[MARKDOWN FIX] Function called. Text length: {len(text)}")
        # logger.info(f"[MARKDOWN FIX] First 100 chars: {repr(text[:100])}")
        # logger.info(f"[MARKDOWN FIX] Contains backslash-n: {(chr(92) + 'n') in text}")
        # logger.info(f"[MARKDOWN FIX] Contains >: {'>' in text}")

        # =====================================================================
        # PHASE 1: Fix literal escaped sequences
        # =====================================================================
        # The model sometimes outputs the literal characters '\' and 'n' instead
        # of actual newlines. We need to detect and replace these.

        # Pattern 1: Literal \n\n (double newline as text)
        # The model outputs the actual two characters: backslash (\) and letter n
        # NOT an actual newline character
        # We need to check for backslash (\\) followed by n
        backslash_n = chr(92) + 'n'  # Explicit: backslash character + letter n
        if backslash_n in text:
            # logger.info(f"[MARKDOWN FIX] Found literal backslash-n, replacing...")
            original_count = text.count(backslash_n)

            # Replace literal \n\n (backslash-n-backslash-n) with actual double newline
            double_backslash_n = backslash_n + backslash_n
            text = text.replace(double_backslash_n, '\n\n')

            # Then replace remaining literal \n (backslash-n) with actual newline
            text = text.replace(backslash_n, '\n')
            # logger.info(f"[MARKDOWN FIX] Replaced {original_count} occurrences of backslash-n")
            # logger.info(f"[MARKDOWN FIX] Text now has {text.count(chr(10))} actual newlines")
        # else:
        #     logger.info(f"[MARKDOWN FIX] No literal backslash-n found")

        # =====================================================================
        # PHASE 2: Normalize Unicode characters and whitespace
        # =====================================================================

        # Normalize Unicode list markers to standard markdown
        # The model sometimes outputs Unicode characters instead of ASCII markdown
        # - U+2013 (–) en-dash → ASCII dash (-)
        # - U+2014 (—) em-dash → ASCII dash (-)
        # - U+2022 (•) bullet point → ASCII dash (-)
        # - U+2023 (‣) triangular bullet → ASCII dash (-)
        text = text.replace('–', '-')  # en-dash
        text = text.replace('—', '-')  # em-dash
        text = text.replace('•', '-')  # bullet point
        text = text.replace('‣', '-')  # triangular bullet

        # Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]  # Remove trailing spaces
        text = '\n'.join(lines)

        # Normalize multiple spaces to single space (but preserve intentional indentation)
        # Only collapse multiple spaces that aren't at the start of a line
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Preserve leading whitespace (indentation)
            leading_spaces = len(line) - len(line.lstrip())
            content = line.lstrip()
            # Collapse multiple spaces in content
            content = re.sub(r' {2,}', ' ', content)
            # Reconstruct line
            normalized_lines.append(' ' * leading_spaces + content)
        text = '\n'.join(normalized_lines)

        # =====================================================================
        # PHASE 3: Fix markdown element spacing
        # =====================================================================

        # Ensure blank line before blockquotes (>)
        # Pattern: any text followed by > should have \n\n before >
        # We need to preserve everything before the >, including punctuation and spaces
        # Match: (any chars that aren't newline)(optional whitespace)(>)
        # Replace with: (captured text)\n\n>
        # Use \s* to match zero or more spaces (handles both "text. >" and "text.>")
        if '>' in text:
            logger.info(f"[MARKDOWN FIX] Found > character, applying blockquote spacing fix...")
            # Find context around >
            idx = text.index('>')
            context_before = text[max(0, idx - 50):idx]
            context_after = text[idx:min(len(text), idx + 50)]
            logger.info(f"[MARKDOWN FIX] Context before >: {repr(context_before)}")
            logger.info(f"[MARKDOWN FIX] Context after >: {repr(context_after)}")

        text = re.sub(r'([^\n]+?)\s*(>)', r'\1\n\n\2', text)

        if '>' in text:
            idx = text.index('>')
            context_after_fix = text[max(0, idx - 50):min(len(text), idx + 50)]
            logger.info(f"[MARKDOWN FIX] After blockquote fix: {repr(context_after_fix)}")

        # Ensure blank line before headers (##)
        # Pattern: any non-newline character followed by # should have \n\n before #
        text = re.sub(r'([^\n])\s*\n?\s*(#{1,6}\s)', r'\1\n\n\2', text)

        # Ensure blank line before unordered lists (* or -)
        # But only if the list starts after regular text (not after another list item)
        text = re.sub(r'([^\n*-])\s*\n\s*([*-]\s+\*\*)', r'\1\n\n\2', text)

        # =====================================================================
        # PHASE 4: Fix list formatting
        # =====================================================================

        # Process line by line to fix list indentation
        lines = text.split('\n')
        fixed_lines = []

        # Track the current list context
        # When we see a * bullet, all following - bullets should be indented
        # until we see another * bullet or non-list content
        in_star_bullet_context = False

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Blank lines always reset star-bullet context so that - lines in a
            # new section are not incorrectly indented as sub-bullets
            if not stripped:
                in_star_bullet_context = False
                fixed_lines.append(line)
                continue

            # Check if this is a star bullet (main list item)
            if stripped.startswith('* '):
                in_star_bullet_context = True
                fixed_lines.append(line)
                continue

            # ## headers also reset star-bullet context
            if stripped.startswith('#'):
                in_star_bullet_context = False
                fixed_lines.append(line)
                continue

            # Check if this is a dash bullet (potential sub-item)
            if stripped.startswith('- '):
                # Only auto-indent genuine sub-bullets: lines that follow a * bullet
                # without a blank line gap AND are not bold-only section titles
                # (bold-only = "- **Title**" with no colon, used as a section divider)
                is_bold_title = (
                        stripped.startswith('- **')
                        and stripped.endswith('**')
                        and ':' not in stripped
                )
                if in_star_bullet_context and not line.startswith('   ') and not is_bold_title:
                    logger.debug(
                        "[MARKDOWN FIX] Auto-indenting dash bullet as sub-item: %r", stripped[:60]
                    )
                    fixed_lines.append('   ' + stripped)
                    continue
                else:
                    fixed_lines.append(line)
                    continue

            # If we encounter non-list content, reset the context
            if not stripped.startswith(('* ', '- ', '>', '#')):
                if i > 0:
                    prev_stripped = lines[i - 1].lstrip()
                    if not prev_stripped.startswith(('* ', '- ')):
                        in_star_bullet_context = False

            fixed_lines.append(line)

        text = '\n'.join(fixed_lines)

        # =====================================================================
        # PHASE 5: Fix punctuation spacing
        # =====================================================================

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Ensure single space after sentence-ending punctuation followed by capital letter
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # Ensure single space after commas/semicolons (not multiple)
        text = re.sub(r'([,;])([^\s])', r'\1 \2', text)

        # =====================================================================
        # PHASE 6: Clean up excessive blank lines
        # =====================================================================

        # Reduce 3+ consecutive newlines to 2 (paragraph break)
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        # =====================================================================
        # PHASE 7: Fix specific markdown quirks
        # =====================================================================

        # Ensure blockquote content is on the same line as >
        # Pattern: > followed by newline and capital letter
        text = re.sub(r'>\s*\n\s*([A-Z])', r'> \1', text)

        # Remove standalone # at the very start (artifact from model)
        # But preserve ## headings
        if text.startswith('#\n'):
            text = text[2:]  # Remove the # and newline

        return text.strip()

    def _OLD_fix_markdown_formatting(self, text: str) -> str:
        """
        Fix common Markdown formatting issues from LLM output.
        
        This function uses a robust, pattern-based approach to detect and fix
        formatting issues without hard-coded rules for specific cases.
        
        Handles:
        1. Literal backslash-n sequences (\\n) appearing in text
        2. Missing blank lines before markdown special elements
        3. Improper spacing around punctuation
        4. List indentation issues
        
        Args:
            text: Generated text with potential formatting issues
            
        Returns:
            Text with corrected Markdown formatting
        """
        import logging
        logger = logging.getLogger(__name__)

        if not text or len(text.strip()) == 0:
            return text

        # Debug: Log function entry
        # logger.info(f"[MARKDOWN FIX] Function called. Text length: {len(text)}")
        # logger.info(f"[MARKDOWN FIX] First 100 chars: {repr(text[:100])}")
        # logger.info(f"[MARKDOWN FIX] Contains backslash-n: {(chr(92) + 'n') in text}")
        # logger.info(f"[MARKDOWN FIX] Contains >: {'>' in text}")

        # =====================================================================
        # PHASE 1: Fix literal escaped sequences
        # =====================================================================
        # The model sometimes outputs the literal characters '\' and 'n' instead
        # of actual newlines. We need to detect and replace these.

        # Pattern 1: Literal \n\n (double newline as text)
        # The model outputs the actual two characters: backslash (\) and letter n
        # NOT an actual newline character
        # We need to check for backslash (\\) followed by n
        backslash_n = chr(92) + 'n'  # Explicit: backslash character + letter n
        if backslash_n in text:
            # logger.info(f"[MARKDOWN FIX] Found literal backslash-n, replacing...")
            original_count = text.count(backslash_n)

            # Replace literal \n\n (backslash-n-backslash-n) with actual double newline
            double_backslash_n = backslash_n + backslash_n
            text = text.replace(double_backslash_n, '\n\n')

            # Then replace remaining literal \n (backslash-n) with actual newline
            text = text.replace(backslash_n, '\n')
            # logger.info(f"[MARKDOWN FIX] Replaced {original_count} occurrences of backslash-n")
            # logger.info(f"[MARKDOWN FIX] Text now has {text.count(chr(10))} actual newlines")
        # else:
        #     logger.info(f"[MARKDOWN FIX] No literal backslash-n found")

        # =====================================================================
        # PHASE 2: Normalize Unicode characters and whitespace
        # =====================================================================

        # Normalize Unicode list markers to standard markdown
        # The model sometimes outputs Unicode characters instead of ASCII markdown
        # - U+2013 (–) en-dash → ASCII dash (-)
        # - U+2014 (—) em-dash → ASCII dash (-)
        # - U+2022 (•) bullet point → ASCII dash (-)
        # - U+2023 (‣) triangular bullet → ASCII dash (-)
        text = text.replace('–', '-')  # en-dash
        text = text.replace('—', '-')  # em-dash
        text = text.replace('•', '-')  # bullet point
        text = text.replace('‣', '-')  # triangular bullet

        # Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]  # Remove trailing spaces
        text = '\n'.join(lines)

        # Normalize multiple spaces to single space (but preserve intentional indentation)
        # Only collapse multiple spaces that aren't at the start of a line
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Preserve leading whitespace (indentation)
            leading_spaces = len(line) - len(line.lstrip())
            content = line.lstrip()
            # Collapse multiple spaces in content
            content = re.sub(r' {2,}', ' ', content)
            # Reconstruct line
            normalized_lines.append(' ' * leading_spaces + content)
        text = '\n'.join(normalized_lines)

        # =====================================================================
        # PHASE 3: Fix markdown element spacing
        # =====================================================================

        # Ensure blank line before blockquotes (>)
        # Pattern: any text followed by > should have \n\n before >
        # We need to preserve everything before the >, including punctuation and spaces
        # Match: (any chars that aren't newline)(optional whitespace)(>)
        # Replace with: (captured text)\n\n>
        # Use \s* to match zero or more spaces (handles both "text. >" and "text.>")
        if '>' in text:
            logger.info(f"[MARKDOWN FIX] Found > character, applying blockquote spacing fix...")
            # Find context around >
            idx = text.index('>')
            context_before = text[max(0, idx - 50):idx]
            context_after = text[idx:min(len(text), idx + 50)]
            logger.info(f"[MARKDOWN FIX] Context before >: {repr(context_before)}")
            logger.info(f"[MARKDOWN FIX] Context after >: {repr(context_after)}")

        text = re.sub(r'([^\n]+?)\s*(>)', r'\1\n\n\2', text)

        if '>' in text:
            idx = text.index('>')
            context_after_fix = text[max(0, idx - 50):min(len(text), idx + 50)]
            logger.info(f"[MARKDOWN FIX] After blockquote fix: {repr(context_after_fix)}")

        # Ensure blank line before headers (##)
        # Pattern: any non-newline character followed by # should have \n\n before #
        text = re.sub(r'([^\n])\s*\n?\s*(#{1,6}\s)', r'\1\n\n\2', text)

        # Ensure blank line before unordered lists (* or -)
        # But only if the list starts after regular text (not after another list item)
        text = re.sub(r'([^\n*-])\s*\n\s*([*-]\s+\*\*)', r'\1\n\n\2', text)

        # =====================================================================
        # PHASE 4: Fix list formatting
        # =====================================================================

        # Process line by line to fix list indentation
        lines = text.split('\n')
        fixed_lines = []

        # Track the current list context
        # When we see a * bullet, all following - bullets should be indented
        # until we see another * bullet or non-list content
        in_star_bullet_context = False

        for i, line in enumerate(lines):
            stripped = line.lstrip()

            # Check if this is a star bullet (main list item)
            if stripped.startswith('* '):
                in_star_bullet_context = True
                fixed_lines.append(line)
                continue

            # Check if this is a dash bullet (potential sub-item)
            if stripped.startswith('- '):
                # If we're in a star bullet context and not already indented, indent it
                if in_star_bullet_context and not line.startswith('   '):
                    fixed_lines.append('   ' + stripped)
                    continue
                else:
                    fixed_lines.append(line)
                    continue

            # If we encounter non-list content, reset the context
            if stripped and not stripped.startswith(('* ', '- ', '>', '#')):
                # But only if it's not a continuation of a list item
                # (list items can have multi-line content)
                # Check if previous line was a list item
                if i > 0:
                    prev_stripped = lines[i - 1].lstrip()
                    # If previous line was not a list item, reset context
                    if not prev_stripped.startswith(('* ', '- ')):
                        in_star_bullet_context = False

            fixed_lines.append(line)

        text = '\n'.join(fixed_lines)

        # =====================================================================
        # PHASE 5: Fix punctuation spacing
        # =====================================================================

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)

        # Ensure single space after sentence-ending punctuation followed by capital letter
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

        # Ensure single space after commas/semicolons (not multiple)
        text = re.sub(r'([,;])([^\s])', r'\1 \2', text)

        # =====================================================================
        # PHASE 6: Clean up excessive blank lines
        # =====================================================================

        # Reduce 3+ consecutive newlines to 2 (paragraph break)
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        # =====================================================================
        # PHASE 7: Fix specific markdown quirks
        # =====================================================================

        # Ensure blockquote content is on the same line as >
        # Pattern: > followed by newline and capital letter
        text = re.sub(r'>\s*\n\s*([A-Z])', r'> \1', text)

        # Remove standalone # at the very start (artifact from model)
        # But preserve ## headings
        if text.startswith('#\n'):
            text = text[2:]  # Remove the # and newline

        return text.strip()

    def _truncate_degenerate_text(self, text: str) -> str:
        """
        Detect and truncate degenerate/rambling text that occurs when LLM
        runs out of substantive content but continues generating.

        Uses multi-signal detection:
        1. Run-on sentences (150+ words without period)
        2. Vocabulary collapse (same words repeating heavily)
        3. Abstract philosophical word density
        4. Punctuation density drop

        Returns:
            Text truncated at the transition point to degenerate content
        """
        if len(text) < 500:
            return text  # Short responses don't degenerate

        # Split into sentences for analysis
        # IMPORTANT: Use a pattern that PRESERVES the whitespace between sentences
        # This ensures we don't destroy markdown formatting (like \n\n before blockquotes)
        sentence_pattern = r'(?<=[.!?])(\s+)'
        parts = re.split(sentence_pattern, text)

        # Reconstruct sentences with their trailing whitespace
        sentences = []
        sentence_whitespace = []
        for i in range(0, len(parts), 2):
            if i < len(parts):
                sentences.append(parts[i])
                # Store the whitespace that follows this sentence
                if i + 1 < len(parts):
                    sentence_whitespace.append(parts[i + 1])
                else:
                    sentence_whitespace.append('')

        if len(sentences) < 3:
            return text

        # Abstract/philosophical words that signal degenerate rambling
        abstract_words = {
            'essence', 'eternal', 'consciousness', 'enlightenment', 'harmony',
            'balance', 'equilibrium', 'symmetry', 'beauty', 'truth', 'goodness',
            'wisdom', 'destiny', 'forevermore', 'everlasting', 'enduring',
            'continuum', 'perpetual', 'transcendent', 'infinite', 'cosmic',
            'universal', 'collective', 'solidarity', 'prosperity', 'wellbeing',
            'vitality', 'dynamism', 'momentum', 'synergy', 'holistic',
            'paradigm', 'zeitgeist', 'manifestation', 'realization', 'actualization',
            'hitherto', 'wherefore', 'herewith', 'aforementioned', 'thereto',
            'coexisting', 'interdependent', 'irrevocably', 'fundamentally',
        }

        # Analyze each sentence for degeneration signals
        degeneration_start_idx = None

        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            word_count = len(words)

            if word_count < 5:
                continue  # Skip very short sentences

            # Signal 1: Run-on sentence (very long without proper structure)
            if word_count > 120:
                logger.warning(f"Degenerate text detected: run-on sentence ({word_count} words) at sentence {i}")
                degeneration_start_idx = i
                break

            # Signal 2: High abstract word density (>10% of words are abstract)
            abstract_count = sum(1 for w in words if w.strip('.,!?;:') in abstract_words)
            abstract_ratio = abstract_count / word_count if word_count > 0 else 0

            if abstract_ratio > 0.10 and word_count > 30:
                logger.warning(
                    f"Degenerate text detected: high abstract word density ({abstract_ratio:.1%}) at sentence {i}")
                degeneration_start_idx = i
                break

            # Signal 3: Vocabulary collapse - check for low unique word ratio
            unique_words = set(w.strip('.,!?;:') for w in words if len(w) > 3)
            unique_ratio = len(unique_words) / word_count if word_count > 0 else 1

            if unique_ratio < 0.4 and word_count > 40:
                logger.warning(
                    f"Degenerate text detected: vocabulary collapse ({unique_ratio:.1%} unique) at sentence {i}")
                degeneration_start_idx = i
                break

            # Signal 4: Missing punctuation in long stretch
            # Check if this sentence + next few have very low punctuation
            if i < len(sentences) - 2:
                combined = ''.join(
                    [sentences[j] + sentence_whitespace[j] for j in range(i, min(i + 3, len(sentences)))])
                combined_words = len(combined.split())
                period_count = combined.count('.') + combined.count('!') + combined.count('?')

                if combined_words > 150 and period_count < 2:
                    logger.warning(f"Degenerate text detected: punctuation drought at sentence {i}")
                    degeneration_start_idx = i
                    break

        # If degeneration detected, truncate at the transition point
        if degeneration_start_idx is not None:
            # Keep all sentences before degeneration
            if degeneration_start_idx > 0:
                # Reconstruct text preserving original whitespace
                valid_parts = []
                for i in range(degeneration_start_idx):
                    valid_parts.append(sentences[i])
                    if i < len(sentence_whitespace):
                        valid_parts.append(sentence_whitespace[i])

                truncated_text = ''.join(valid_parts).rstrip()

                # Ensure it ends with proper punctuation
                if truncated_text and truncated_text[-1] not in '.!?':
                    truncated_text += '.'

                logger.warning(
                    f"Truncated degenerate text: kept {degeneration_start_idx}/{len(sentences)} sentences "
                    f"({len(truncated_text)}/{len(text)} chars)"
                )
                return truncated_text
            else:
                # Degeneration from the start - something is very wrong
                # Return first 500 chars as fallback
                logger.error("Text appears degenerate from start - returning truncated fallback")
                fallback = text[:500]
                last_period = fallback.rfind('.')
                if last_period > 100:
                    return fallback[:last_period + 1]
                return fallback

        return text

    def generate_batch(
            self,
            prompts: List[str],
            **kwargs
    ) -> List[str]:
        """
        Generate answers for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Generation parameters

        Returns:
            List of generated answers
        """
        logger.info(f"Generating batch of {len(prompts)} answers")

        answers = []
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"   Processing {i}/{len(prompts)}")
            answer = self.generate(prompt, **kwargs)
            answers.append(answer)

        logger.info("Batch generation complete")
        return answers

    def generate_with_metadata(
            self,
            prompt: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with metadata.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters

        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        answer = self.generate(prompt, **kwargs)

        elapsed_time = time.time() - start_time
        # Estimate tokens (vLLM doesn't expose tokenizer)
        output_tokens = int(len(answer.split()) * 1.3)
        input_tokens = int(len(prompt.split()) * 1.3)

        return {
            "answer": answer,
            "generation_time": elapsed_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / elapsed_time if elapsed_time > 0 else 0,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_new_tokens
        }

    def generate_with_citations(
            self,
            prompt: str,
            source_documents: List[Dict[str, Any]],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer with automatic citation extraction and validation.

        This method combines generation with post-processing to extract and
        validate citations, producing structured output ready for display.

        Args:
            prompt: Input prompt with context
            source_documents: List of source documents used in context
            **kwargs: Generation parameters

        Returns:
            Dictionary with:
            - answer: Clean answer text with valid citations
            - citations: List of citation objects with metadata
            - metadata: Generation metadata
            - validation: Citation validation results
        """
        from src.generation.citation_extractor import CitationExtractor

        logger.info("Generating answer with citations")

        # Generate answer
        start_time = time.time()
        answer = self.generate(prompt, **kwargs)
        generation_time = time.time() - start_time

        # Extract and validate citations
        extractor = CitationExtractor()
        citations, valid_ids, invalid_ids = extractor.extract_citations(
            answer,
            source_documents
        )

        # Remove invalid citations from answer
        clean_answer = extractor.remove_invalid_citations(answer, invalid_ids)

        # Validate citation coverage
        coverage = extractor.validate_citation_coverage(clean_answer)

        # Calculate confidence score (simple version)
        confidence = self._calculate_simple_confidence(
            clean_answer,
            citations,
            coverage,
            source_documents
        )

        # Format output
        result = {
            "answer": clean_answer,
            "citations": [cit.to_dict() for cit in citations],
            "metadata": {
                "generation_time": round(generation_time, 2),
                "input_tokens": int(len(prompt.split()) * 1.3),
                "output_tokens": int(len(clean_answer.split()) * 1.3),
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "confidence": confidence
            },
            "validation": {
                "total_citations": len(citations),
                "valid_citations": len(valid_ids),
                "invalid_citations": len(invalid_ids),
                "coverage": coverage
            }
        }

        logger.info(f"   Generated answer with {len(citations)} valid citations")
        logger.info(f"   Confidence: {confidence:.3f}")

        return result

    @staticmethod
    def _calculate_simple_confidence(
            answer: str,
            citations: List[Any],
            coverage: Dict[str, Any],
            source_documents: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate a simple confidence score for the generated answer.

        Args:
            answer: Generated answer text
            citations: Extracted citations
            coverage: Citation coverage metrics
            source_documents: Source documents

        Returns:
            Confidence score (0.0-1.0)
        """
        # Citation coverage score (30%)
        coverage_score = coverage.get('coverage', 0.0)

        # Source quality score (30%) - based on rerank scores
        if citations:
            avg_relevance = sum(cit.relevance_score for cit in citations) / len(citations)
            source_quality = avg_relevance
        else:
            source_quality = 0.0

        # Answer completeness (20%) - has answer and reasonable length
        min_length = 50
        completeness = min(1.0, len(answer) / min_length) if answer else 0.0

        # Citation ratio (20%) - good balance of citations
        words = len(answer.split())
        citations_count = len(citations)
        if words > 0:
            # Ideal: 1 citation per 30-50 words
            ideal_ratio = words / 40
            citation_ratio = min(1.0, citations_count / ideal_ratio) if ideal_ratio > 0 else 0.0
        else:
            citation_ratio = 0.0

        # Weighted combination
        confidence = (
                coverage_score * 0.30 +
                source_quality * 0.30 +
                completeness * 0.20 +
                citation_ratio * 0.20
        )

        return round(confidence, 3)

    def get_config(self) -> Dict[str, Any]:
        """Get generator configuration."""
        return {
            "model_name": self.config.model_name,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "device": self.config.device,
            "timeout": self.config.timeout
        }


def create_llm_generator(
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        **kwargs
) -> LLMGenerator:
    """
    Factory function to create LLMGenerator.

    Args:
        model_name: Model name or path
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLMGenerator instance
    """
    config = GenerationConfig(model_name=model_name, **kwargs)
    return LLMGenerator(config=config)
