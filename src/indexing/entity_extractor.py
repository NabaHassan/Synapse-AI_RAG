"""
Entity Extractor Module for Enhanced Document Indexing.

Extracts named entities (PERSON, ORG, GPE, DATE, LAW, MONEY) from text
using configurable NER backends:
  - spaCy (en_core_web_sm / en_core_web_lg)
  - Transformer-based (dslim/bert-base-NER via HuggingFace transformers)

Provides both document-level entity counts and chunk-level entity lists
for storage in Qdrant payloads.
"""

import re
import logging
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Supported entity types for extraction
SUPPORTED_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "LAW", "MONEY", "NORP", "FAC"}

# Mapping from transformer NER labels to unified types
TRANSFORMER_LABEL_MAP = {
    "PER": "PERSON",
    "B-PER": "PERSON",
    "I-PER": "PERSON",
    "ORG": "ORG",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "LOC": "GPE",
    "B-LOC": "GPE",
    "I-LOC": "GPE",
    "MISC": "MISC",
    "B-MISC": "MISC",
    "I-MISC": "MISC",
}


class EntityExtractor:
    """
    Extracts named entities from text using configurable NER backends.

    Supports:
        - backend="spacy": Uses spaCy NER models (default: en_core_web_lg)
        - backend="transformer": Uses dslim/bert-base-NER via HuggingFace

    Usage:
        extractor = EntityExtractor(backend="spacy", model_name="en_core_web_lg")
        doc_entities = extractor.extract_from_text("John Doe met with Acme Corp...")
        chunk_entities = extractor.extract_from_chunk("John Doe was present...")
    """

    def __init__(
            self,
            backend: str = "spacy",
            model_name: Optional[str] = None,
            entity_types: Optional[set] = None
    ):
        """
        Initialize the entity extractor.

        Args:
            backend: NER backend - "spacy" or "transformer"
            model_name: Model name. Defaults:
                - spaCy: "en_core_web_lg"
                - transformer: "dslim/bert-base-NER"
            entity_types: Set of entity types to extract.
                Defaults to SUPPORTED_ENTITY_TYPES.
        """
        self.backend = backend.lower()
        self.entity_types = entity_types or SUPPORTED_ENTITY_TYPES
        self._nlp = None
        self._ner_pipeline = None

        if self.backend == "spacy":
            self.model_name = model_name or "en_core_web_lg"
            self._init_spacy()
        elif self.backend == "transformer":
            self.model_name = model_name or "dslim/bert-base-NER"
            self._init_transformer()
        else:
            raise ValueError(
                f"Unsupported NER backend: '{backend}'. Use 'spacy' or 'transformer'."
            )

        logger.info(f"EntityExtractor initialized (backend={self.backend}, model={self.model_name})")

    def _init_spacy(self):
        """Initialize spaCy NER model."""
        try:
            import spacy
            self._nlp = spacy.load(self.model_name)
            logger.info(f"  Loaded spaCy model: {self.model_name}")
        except OSError:
            logger.error(
                f"spaCy model '{self.model_name}' not found. "
                f"Install it with: python -m spacy download {self.model_name}"
            )
            raise
        except ImportError:
            logger.error("spaCy is not installed. Install with: pip install spacy")
            raise

    def _init_transformer(self):
        """Initialize HuggingFace transformer NER pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            self._ner_pipeline = hf_pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple"
            )
            logger.info(f"  Loaded transformer NER model: {self.model_name}")
        except ImportError:
            logger.error(
                "transformers library not installed. "
                "Install with: pip install transformers torch"
            )
            raise

    # =========================================================================
    # Public API
    # =========================================================================

    def extract_from_text(self, text: str) -> Dict:
        """
        Extract entities from a full document text.

        Returns document-level entity counts and a list of unique entities
        with their types and total occurrence counts.

        Args:
            text: Full document text

        Returns:
            {
                "entities": [
                    {"name": "John Doe", "type": "PERSON", "count": 5},
                    {"name": "Acme Corp", "type": "ORG", "count": 3},
                    ...
                ],
                "entity_counts": {
                    "John Doe": 5,
                    "Acme Corp": 3,
                    ...
                }
            }
        """
        if not text or not text.strip():
            return {"entities": [], "entity_counts": {}}

        if self.backend == "spacy":
            raw_entities = self._extract_spacy(text)
        else:
            raw_entities = self._extract_transformer(text)

        # Count occurrences
        entity_counter = Counter()
        entity_type_map = {}

        for ent_name, ent_type in raw_entities:
            normalized = self._normalize_entity_name(ent_name)
            if not normalized or len(normalized) < 2:
                continue
            entity_counter[normalized] += 1
            # Keep the most common type for each entity
            if normalized not in entity_type_map:
                entity_type_map[normalized] = ent_type

        # Build result
        entities_list = []
        entity_counts = {}

        for name, count in entity_counter.most_common():
            entities_list.append({
                "name": name,
                "type": entity_type_map.get(name, "UNKNOWN"),
                "count": count
            })
            entity_counts[name] = count

        return {
            "entities": entities_list,
            "entity_counts": entity_counts
        }

    def extract_from_chunk(self, chunk_text: str) -> List[Dict]:
        """
        Extract entities from a single chunk of text.

        Returns a list of entities found in this chunk with their mention
        counts within the chunk.

        Args:
            chunk_text: Text of a single chunk

        Returns:
            [
                {"name": "John Doe", "type": "PERSON", "mention_count_in_chunk": 2},
                {"name": "Acme Corp", "type": "ORG", "mention_count_in_chunk": 1},
                ...
            ]
        """
        if not chunk_text or not chunk_text.strip():
            return []

        if self.backend == "spacy":
            raw_entities = self._extract_spacy(chunk_text)
        else:
            raw_entities = self._extract_transformer(chunk_text)

        # Count per-chunk occurrences
        entity_counter = Counter()
        entity_type_map = {}

        for ent_name, ent_type in raw_entities:
            normalized = self._normalize_entity_name(ent_name)
            if not normalized or len(normalized) < 2:
                continue
            entity_counter[normalized] += 1
            if normalized not in entity_type_map:
                entity_type_map[normalized] = ent_type

        # Build result
        chunk_entities = []
        for name, count in entity_counter.most_common():
            chunk_entities.append({
                "name": name,
                "type": entity_type_map.get(name, "UNKNOWN"),
                "mention_count_in_chunk": count
            })

        return chunk_entities

    # =========================================================================
    # Backend-specific extraction
    # =========================================================================

    def _extract_spacy(self, text: str) -> List[tuple]:
        """
        Extract entities using spaCy.

        Returns list of (entity_name, entity_type) tuples.
        """
        # spaCy can struggle with very long texts; process in chunks if needed
        max_length = 1_000_000  # spaCy's default max is 1M chars
        if len(text) > max_length:
            logger.warning(
                f"Text length ({len(text)}) exceeds spaCy max. "
                f"Processing first {max_length} characters."
            )
            text = text[:max_length]

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append((ent.text, ent.label_))

        return entities

    def _extract_transformer(self, text: str) -> List[tuple]:
        """
        Extract entities using HuggingFace transformer NER.

        Returns list of (entity_name, unified_entity_type) tuples.
        """
        # Transformer models have token limits; process in chunks
        max_chunk_size = 450  # tokens ~= words, safe limit for BERT
        words = text.split()
        entities = []

        for i in range(0, len(words), max_chunk_size):
            chunk = " ".join(words[i:i + max_chunk_size])
            try:
                ner_results = self._ner_pipeline(chunk)
                for result in ner_results:
                    label = result.get("entity_group", result.get("entity", ""))
                    mapped_type = TRANSFORMER_LABEL_MAP.get(label, label)
                    if mapped_type in self.entity_types:
                        word = result.get("word", "").strip()
                        # Clean up subword tokenization artifacts
                        word = word.replace(" ##", "").replace("##", "")
                        if word:
                            entities.append((word, mapped_type))
            except Exception as e:
                logger.warning(f"Transformer NER failed on chunk: {e}")
                continue

        return entities

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """
        Normalize entity name for consistent storage and matching.

        - Strips whitespace
        - Collapses internal whitespace
        - Removes leading/trailing punctuation (but preserves internal like "A. de Rothschild")
        """
        if not name:
            return ""

        # Strip and collapse whitespace
        normalized = re.sub(r'\s+', ' ', name.strip())

        # Remove leading/trailing punctuation (keep internal dots, hyphens)
        normalized = normalized.strip('.,;:!?()[]{}"\'/\\')

        return normalized

    def get_info(self) -> Dict:
        """Return extractor configuration info."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "entity_types": sorted(self.entity_types),
        }
