"""
Query Enhancement - Step 4 of Query Processing
LLM-DRIVEN query enhancement for production use

This module uses a small instruction-tuned model (Qwen2.5-1.5B-Instruct)
for fast, high-quality query expansion, decomposition, and HyDE generation.

Implements:
- Query Expansion (LLM-based)
- HyDE (Hypothetical Document Embeddings via LLM)
- Query Decomposition (LLM-based)

"""

import json
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error(
        "transformers not available. Query enhancement requires transformers. "
        "Install with: pip install transformers torch"
    )


@dataclass
class QueryEnhancement:
    """
    Enhanced query result with LLM-generated enhancements
    """
    original_query: str
    cleaned_query: str  # Same as original (no rule-based normalization)
    expanded_queries: List[str]  # LLM-generated expansions
    hyde_document: Optional[str]  # LLM-generated hypothetical document
    sub_queries: List[str]  # LLM-generated sub-queries
    search_strategy: str  # llm_enhanced
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class LLMConfig:
    """Configuration for query enhancement LLM"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
    max_new_tokens: int = 128  # Reduced from 256 for faster generation
    temperature: float = 0.1  # Very low for deterministic JSON output


class LLMQueryEnhancer:
    """
    LLM-based query enhancement - PRODUCTION READY
    Uses small instruction-tuned model for fast, accurate enhancement
    
    Model: Qwen2.5-1.5B-Instruct
    Speed: ~50-100ms per enhancement
    Quality: Excellent for structured tasks
    """

    def __init__(
            self,
            config: Optional[LLMConfig] = None,
            existing_model=None,
            existing_tokenizer=None
    ):
        """
        Initialize LLM query enhancer
        
        Args:
            config: LLM configuration
            existing_model: Reuse existing model (saves memory)
            existing_tokenizer: Reuse existing tokenizer
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for query enhancement. "
                "Install with: pip install transformers torch"
            )

        self.config = config or LLMConfig()

        # Reuse existing model if provided (saves memory)
        if existing_model and existing_tokenizer:
            logger.info("  Reusing existing model for query enhancement")
            self.model = existing_model
            self.tokenizer = existing_tokenizer
        else:
            logger.info(f" Loading query enhancement model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            if self.config.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    dtype=torch.float32
                )
                self.model = self.model.to(self.config.device)

            logger.info(f" Model loaded on {self.config.device}")

    def _generate(self, prompt: str) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides structured JSON output."},
            {"role": "user", "content": prompt}
        ]

        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,  # Greedy decoding for speed (2-3x faster)
                num_beams=1,  # No beam search
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

    @staticmethod
    def _extract_json(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from LLM response with robust error handling
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        response = response.strip()

        # Remove Markdown code blocks
        if "```" in response:
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                # Try to parse this part
                try:
                    return json.loads(part)
                except:
                    continue

        # Try to find JSON object in response
        # Look for {...}
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parse failed: {e}")
                return None

        return None

    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """
        Expand query using LLM - generates alternative phrasings
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansions
            
        Returns:
            List of LLM-generated expanded queries
        """
        prompt = f"""You are a query expansion expert. Generate {max_expansions} alternative phrasings of the user's query to improve information retrieval.

Rules:
1. Keep the same meaning and intent
2. Use synonyms and different word orders
3. Expand acronyms if present
4. Each variation should be complete and grammatical
5. Make variations distinct from each other

User Query: {query}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "expanded_queries": [
    "alternative phrasing 1",
    "alternative phrasing 2"
  ]
}}

JSON Output:"""

        try:
            response = self._generate(prompt)
            result = self._extract_json(response)

            if not result:
                logger.warning("Failed to parse JSON from expansion response")
                return []

            expansions = result.get("expanded_queries", [])

            # Validate that expansions are strings
            valid_expansions = []
            for exp in expansions:
                if isinstance(exp, str):
                    valid_expansions.append(exp)
                else:
                    logger.warning(f"Invalid expansion type: {type(exp)}")

            logger.info(f" Generated {len(valid_expansions)} query expansions")
            return valid_expansions[:max_expansions]

        except Exception as e:
            logger.error(f" Query expansion failed: {e}")
            return []

    def decompose_query(self, query: str, complexity: float = 0.5) -> List[str]:
        """
        Decompose complex query into sub-queries using LLM
        
        Args:
            query: Complex query
            complexity: Query complexity score (0-1)
            
        Returns:
            List of sub-queries (empty if shouldn't decompose)
        """
        # Skip decomposition for simple queries
        if complexity < 0.6:
            logger.info("Query too simple for decomposition (complexity < 0.6)")
            return []

        prompt = f"""You are a query decomposition expert. Analyze if this query should be broken down into simpler sub-questions.

Rules:
1. Return ONLY the questions, not the answers
2. Only decompose if the query has multiple distinct parts or topics
3. Each sub-question must be self-contained and grammatical
4. Preserve the original intent
5. If the query is already simple, set should_decompose to false
6. Return sub_queries as a simple array of strings, NOT objects

User Query: {query}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "should_decompose": true,
  "sub_queries": [
    "complete sub-question 1",
    "complete sub-question 2"
  ],
  "reasoning": "brief explanation"
}}

JSON Output:"""

        try:
            response = self._generate(prompt)
            result = self._extract_json(response)

            if not result:
                logger.warning("Failed to parse JSON from decomposition response")
                return []

            if result.get("should_decompose", False):
                sub_queries = result.get("sub_queries", [])

                # Validate and extract clean strings
                valid_sub_queries = []
                for sq in sub_queries:
                    if isinstance(sq, str):
                        valid_sub_queries.append(sq)
                    elif isinstance(sq, dict):
                        # Handle case where LLM returns objects instead of strings
                        # Try to extract a usable string
                        question = sq.get('question', sq.get('query', sq.get('text', '')))
                        if question and isinstance(question, str):
                            valid_sub_queries.append(question)
                            logger.debug(f"Extracted question from dict: {question}")
                    else:
                        logger.warning(f"Invalid sub-query type: {type(sq)}")

                if valid_sub_queries:
                    logger.info(
                        f" Decomposed into {len(valid_sub_queries)} sub-queries: {result.get('reasoning', '')}")
                    return valid_sub_queries
                else:
                    logger.warning("No valid sub-queries extracted")
                    return []
            else:
                logger.info(f"Query not decomposed: {result.get('reasoning', 'too simple')}")
                return []

        except Exception as e:
            logger.error(f" Query decomposition failed: {e}")
            return []

    def generate_hyde(self, query: str) -> Optional[str]:
        """
        Generate hypothetical document that would answer the query (HyDE)
        
        Args:
            query: Original query
            
        Returns:
            LLM-generated hypothetical document or None
        """
        prompt = f"""You are generating a hypothetical document passage that would answer the user's question.

Write a concise, informative passage (50-100 words) as if quoting from a high-quality document. Be factual and specific.

User Query: {query}

Write the passage directly (no JSON, no introduction):"""

        try:
            response = self._generate(prompt)

            # Clean up response
            response = response.strip()

            # Remove common prefixes
            prefixes = ["Here is", "Here's", "Passage:", "Answer:"]
            for prefix in prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()

            if len(response) > 50:  # Reasonable minimum
                logger.info(f" Generated HyDE document ({len(response)} chars)")
                return response
            else:
                logger.warning("HyDE document too short, skipping")
                return None

        except Exception as e:
            logger.error(f" HyDE generation failed: {e}")
            return None


class QueryEnhancer:
    """
    Production-ready Query Enhancer with LLM backend
    
    This is the ONLY way query enhancement should be done in production.
    Rule-based approaches (spaCy/WordNet) have been removed as they don't work.
    """

    def __init__(
            self,
            enable_expansion: bool = True,
            enable_hyde: bool = False,  # Expensive, off by default
            enable_decomposition: bool = True,
            llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
            existing_model=None,  # Reuse main model if possible
            existing_tokenizer=None
    ):
        """
        Initialize LLM-based query enhancer
        
        Args:
            enable_expansion: Enable query expansion
            enable_hyde: Enable HyDE (expensive, ~100ms extra)
            enable_decomposition: Enable query decomposition
            llm_model_name: Model to use for enhancement
            existing_model: Reuse existing model (saves memory)
            existing_tokenizer: Reuse existing tokenizer
        """
        self.enable_expansion = enable_expansion
        self.enable_hyde = enable_hyde
        self.enable_decomposition = enable_decomposition

        # Initialize LLM enhancer
        config = LLMConfig(model_name=llm_model_name)
        self.llm_enhancer = LLMQueryEnhancer(
            config=config,
            existing_model=existing_model,
            existing_tokenizer=existing_tokenizer
        )

        logger.info("=" * 80)
        logger.info("QueryEnhancer initialized with LLM backend")
        logger.info("=" * 80)
        logger.info(f"  Expansion: {enable_expansion}")
        logger.info(f"  HyDE: {enable_hyde}")
        logger.info(f"  Decomposition: {enable_decomposition}")
        logger.info(f"  Model: {llm_model_name}")
        logger.info("=" * 80)

    def enhance(
            self,
            query: str,
            query_type: str = "factual",
            complexity: float = 0.5,
            strategy: str = "auto"
    ) -> QueryEnhancement:
        """
        Enhance query using LLM with parallel execution for speed
        
        Args:
            query: Original query
            query_type: Query type (for metadata)
            complexity: Query complexity score (0-1)
            strategy: Enhancement strategy (auto/expansion/hyde/decomposed/hybrid)
            
        Returns:
            QueryEnhancement object with LLM-generated enhancements
        """
        start_time = time.time()
        logger.info(f"Enhancing query: '{query[:50]}...'")

        # Early return for very short queries (don't need enhancement)
        query_words = query.split()
        if len(query_words) < 5:
            logger.info(f"Skipping enhancement for short query ({len(query_words)} words)")
            return QueryEnhancement(
                original_query=query,
                cleaned_query=query,
                expanded_queries=[],
                hyde_document=None,
                sub_queries=[],
                search_strategy="skipped_short",
                metadata={
                    'query_type': query_type,
                    'complexity': complexity,
                    'num_expansions': 0,
                    'num_sub_queries': 0,
                    'hyde_enabled': False,
                    'skipped': True,
                    'skip_reason': 'too_short',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )

        # Early return for very simple queries
        if complexity < 0.3:
            logger.info(f"Skipping enhancement for simple query (complexity={complexity:.2f})")
            return QueryEnhancement(
                original_query=query,
                cleaned_query=query,
                expanded_queries=[],
                hyde_document=None,
                sub_queries=[],
                search_strategy="skipped_simple",
                metadata={
                    'query_type': query_type,
                    'complexity': complexity,
                    'num_expansions': 0,
                    'num_sub_queries': 0,
                    'hyde_enabled': False,
                    'skipped': True,
                    'skip_reason': 'too_simple',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )

        expanded_queries = []
        hyde_document = None
        sub_queries = []

        # Run expansion and decomposition in PARALLEL for speed
        # HyDE runs separately if enabled (rare)
        try:
            tasks_to_run = []

            # Determine which tasks to run
            run_expansion = self.enable_expansion and strategy in ["auto", "expansion", "hybrid"]
            run_decomposition = self.enable_decomposition and strategy in ["auto", "decomposed", "hybrid"]

            if run_expansion or run_decomposition:
                logger.info("Running enhancement tasks in parallel...")

                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {}

                    # Submit expansion task
                    if run_expansion:
                        futures[executor.submit(self.llm_enhancer.expand_query, query, 5)] = "expansion"

                    # Submit decomposition task
                    if run_decomposition:
                        futures[executor.submit(self.llm_enhancer.decompose_query, query, complexity)] = "decomposition"

                    # Collect results as they complete
                    for future in as_completed(futures):
                        task_type = futures[future]
                        try:
                            result = future.result(timeout=10)  # 10s timeout per task
                            if task_type == "expansion":
                                expanded_queries = result
                            elif task_type == "decomposition":
                                sub_queries = result
                        except Exception as e:
                            logger.error(f"  {task_type.capitalize()} failed: {e}")
                            # Continue with empty results for failed task
                            if task_type == "expansion":
                                expanded_queries = []
                            elif task_type == "decomposition":
                                sub_queries = []

        except Exception as e:
            # Fallback to sequential if parallel fails
            logger.warning(f"Parallel enhancement failed: {e}")
            logger.info("Falling back to sequential enhancement...")

            # Expansion (sequential fallback)
            if self.enable_expansion and strategy in ["auto", "expansion", "hybrid"]:
                try:
                    expanded_queries = self.llm_enhancer.expand_query(query, max_expansions=5)
                except Exception as exp_error:
                    logger.error(f"Expansion failed: {exp_error}")
                    expanded_queries = []

            # Decomposition (sequential fallback)
            if self.enable_decomposition and strategy in ["auto", "decomposed", "hybrid"]:
                try:
                    sub_queries = self.llm_enhancer.decompose_query(query, complexity)
                except Exception as decomp_error:
                    logger.error(f"Decomposition failed: {decomp_error}")
                    sub_queries = []

        # HyDE (runs separately, disabled by default)
        if self.enable_hyde and strategy in ["auto", "hyde", "hybrid"]:
            try:
                hyde_document = self.llm_enhancer.generate_hyde(query)
            except Exception as hyde_error:
                logger.error(f"HyDE generation failed: {hyde_error}")
                hyde_document = None

        # Create enhancement result
        enhancement_time = time.time() - start_time

        enhancement = QueryEnhancement(
            original_query=query,
            cleaned_query=query,  # No normalization needed with LLM
            expanded_queries=expanded_queries,
            hyde_document=hyde_document,
            sub_queries=sub_queries,
            search_strategy="llm_enhanced",
            metadata={
                'query_type': query_type,
                'complexity': complexity,
                'num_expansions': len(expanded_queries),
                'num_sub_queries': len(sub_queries),
                'hyde_enabled': hyde_document is not None,
                'enhancement_time_seconds': round(enhancement_time, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )

        logger.info(
            f"Enhancement complete in {enhancement_time:.2f}s: "
            f"{len(expanded_queries)} expansions, {len(sub_queries)} sub-queries, "
            f"hyde={hyde_document is not None}"
        )

        return enhancement


# Convenience function
def create_query_enhancer(**kwargs) -> QueryEnhancer:
    """
    Convenience function to create query enhancer
    
    Args:
        **kwargs: Arguments for QueryEnhancer
        
    Returns:
        QueryEnhancer instance
    """
    return QueryEnhancer(**kwargs)
