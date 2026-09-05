"""
Anchor Term Extractor for Knowledge Base.

Uses an LLM to extract key domain terms, acronyms, and important concepts
from a set of documents to populate the collection_anchor_terms for a KB.
"""

import logging
from typing import List, Optional
from src.generation.llm_generator import LLMGenerator, GenerationConfig

logger = logging.getLogger(__name__)

class AnchorTermExtractor:
    """
    Extracts key domain terms from text using an LLM.
    """

    def __init__(self, llm_generator: Optional[LLMGenerator] = None):
        """
        Initialize the extractor.
        """
        if llm_generator:
            self.llm = llm_generator
        else:
            # Use the 30B model for extraction
            config = GenerationConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
                temperature=0.1,
                max_new_tokens=512
            )
            self.llm = LLMGenerator(config=config)

    def extract_terms(self, text: str, max_terms: int = 15) -> List[str]:
        """
        Extract key domain terms from a given text.
        """
        if isinstance(text, list):
            text = "\n\n".join([str(t) for t in text])

        if not text or not text.strip():
            return []

        # Construct prompt
        prompt = f"""You are an expert domain analyst. Your task is to extract a highly specific and concise list of 3 key domain terms from the provided text.

These terms will be used to anchor web searches related to this knowledge base. They must be the most critical "keywords" that define the core domain.

Rules:
1. Provide exactly 3 terms. No more, no less.
2. Each term should be 1-2 words long.
3. Be extremely strict: only include terms that are central to the entire document set.
4. Return ONLY a comma-separated list of terms.
5. Do NOT include introductory or concluding text.

### Text:
{text[:8000]}  # Limit text to 8000 chars for prompt safety

### Key Domain Terms:"""

        try:
            response = self.llm.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.1,  # Low temperature for consistency
                purpose="anchor_term_extraction"
            )
            
            # Parse the response
            terms = [t.strip() for t in response.split(",") if t.strip()]
            
            # Clean and filter terms
            cleaned_terms = []
            seen = set()
            for term in terms:
                # Basic cleaning
                term = term.strip(" .-*\"'").lower()
                if term and term not in seen:
                    cleaned_terms.append(term)
                    seen.add(term)
            
            return cleaned_terms[:max_terms]

        except Exception as e:
            logger.error(f"Failed to extract anchor terms: {e}")
            return []

    def extract_scope_description(self, text: str) -> str:
        """
        Extract domain metadata and format into a hardcoded scope template.
        """
        if isinstance(text, list):
            text = "\n\n".join([str(t) for t in text])

        if not text or not text.strip():
            return ""

        # Construct a structured extraction prompt
        prompt = f"""You are an expert domain analyst. Analyze the provided text samples and identify the following three components:
1. PRIMARY TOPICS: The core subjects covered.
2. INTENDED AUDIENCE: Who is this information for?
3. EXCLUDED TOPICS: What subjects are clearly NOT covered?

Rules:
- Provide the answer as three lines starting with "TOPICS:", "AUDIENCE:", and "EXCLUSIONS:".
- Keep each answer to a concise list of keywords or phrases.
- If a component is not apparent, leave it blank (e.g., "EXCLUSIONS: ").

### Text Samples:
{text[:8000]}

### Metadata Extraction:"""

        try:
            response = self.llm.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.1,
                purpose="scope_metadata_extraction"
            )
            
            # Parse the structured response
            metadata = {"topics": "", "audience": "", "exclusions": ""}
            for line in response.split('\n'):
                if line.startswith("TOPICS:"):
                    metadata["topics"] = line.replace("TOPICS:", "").strip()
                elif line.startswith("AUDIENCE:"):
                    metadata["audience"] = line.replace("AUDIENCE:", "").strip()
                elif line.startswith("EXCLUSIONS:"):
                    metadata["exclusions"] = line.replace("EXCLUSIONS:", "").strip()

            # Construct the hardcoded paragraph
            # We only include segments that have content
            scope_parts = []
            if metadata["topics"]:
                scope_parts.append(f"This knowledge base covers {metadata['topics']}.")
            if metadata["audience"]:
                scope_parts.append(f"It is intended for {metadata['audience']}.")
            if metadata["exclusions"]:
                scope_parts.append(f"It does NOT cover {metadata['exclusions']}.")
            
            # Mandatory grounding rule: forces refusal of out-of-scope queries
            scope_parts.append("If a query is not related to this scope, you must state that you do not have that information and cannot answer.")
            
            return " ".join(scope_parts)

        except Exception as e:
            logger.error(f"Failed to extract scope metadata: {e}")
            return ""


    def extract_from_documents(self, documents: List[any], max_terms: int = 15) -> List[str]:
        """
        Extract anchor terms from a list of documents.
        Samples the documents to get a representative text.
        """
        if not documents:
            return []
            
        # Combine content from documents (sampling if many)
        sample_size = min(len(documents), 20)
        # Sort by length descending to get most informative docs first? 
        # Or just take a spread.
        content_samples = []
        for i in range(0, len(documents), max(1, len(documents) // sample_size)):
            doc = documents[i]
            content = getattr(doc, "content", str(doc))
            content_samples.append(content[:1000]) # Take first 1000 chars of each
            
        combined_text = "\n\n".join(content_samples)
        return self.extract_terms(combined_text, max_terms=max_terms)
