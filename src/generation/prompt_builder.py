"""
Prompt Builder for RAG Pipeline.

This module provides prompt construction using Haystack's PromptBuilder
with Jinja2 templating for LLM generation.

Features:
- Dynamic context insertion with citations
- Clear instructions for LLM
- Support for multiple documents
- Flexible template customization
"""

import logging
from haystack import Document
from typing import List, Dict, Any, Optional
from haystack.components.builders import PromptBuilder as HaystackPromptBuilder

logger = logging.getLogger(__name__)

# Default RAG prompt template with citations
DEFAULT_RAG_TEMPLATE = """You are a helpful assistant. Use the provided context to answer the question.
If the answer isn't in the context, say so.

Context:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
(Source: {{ doc.meta.source }}, {% if doc.meta.page %}Page: {{ doc.meta.page }}{% else %}No page info{% endif %})

{% endfor %}

Question: {{ query }}

Instructions:
- Answer based ONLY on the provided context
- ALWAYS cite your sources using [1], [2], [3] etc. immediately after each claim
- Example: "Machine learning is useful [1]." or "Research shows benefits [1][2]."
- Do NOT use [N] as a placeholder - use actual numbers like [1], [2]
- If information is missing, state: "The provided context does not contain this information."
- Be concise and factual
- Stop after giving your answer - do not add notes or repeat instructions

Answer:"""

DEFAULT_RAG_TEMPLATE_3 = """You are a helpful assistant. Use the provided context to answer the question.
If the answer isn't in the context, say so.

Context:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
(Source: {{ doc.meta.source }}, {% if doc.meta.page %}Page: {{ doc.meta.page }}{% else %}No page info{% endif %})
{% endfor %}

Question: {{ query }}

Instructions:
- Provide a detailed, well-structured answer based ONLY on the provided context
- Use markdown formatting: **bold** for key terms, bullet points for lists, ## headings for sections
- Use new lines to separate paragraphs and sections
- Always use real newlines — never output "\n" literally
- Use numbered lists for sequential information
- Organize complex topics with clear headings and subheadings
- ALWAYS cite sources [1], [2], [3] immediately after claims
- Answer ONLY based on provided context
- If information is missing, state: "The provided context does not contain this information."
- Be comprehensive and professional
- Stop after giving your answer
- Always write complete sentences and paragraphs
- Do NOT add extra lines like “Sources cited throughout” or commentary about lacking references.
- Do NOT output notes or disclaimers.

Answer:"""


class RAGPromptBuilder:
    """
    Wrapper for Haystack PromptBuilder with RAG-specific templates.

    Builds prompts for LLM generation with context and citations.
    Uses Jinja2 templating for flexible prompt construction.
    """

    def __init__(
            self,
            template: Optional[str] = None,
            required_variables: Optional[List[str]] = None
    ):
        """
        Initialize the RAG Prompt Builder.

        Args:
            template: Custom Jinja2 template string (default: DEFAULT_RAG_TEMPLATE)
            required_variables: List of required template variables (default: ["query", "documents"])
        """
        self.template = template or DEFAULT_RAG_TEMPLATE_3
        self.required_variables = required_variables or ["query", "documents"]

        logger.info("Initializing RAGPromptBuilder")

        # Initialize Haystack PromptBuilder
        self.haystack_builder = HaystackPromptBuilder(
            template=self.template,
            required_variables=self.required_variables
        )

        logger.info("RAGPromptBuilder initialized successfully")

    def build_prompt(
            self,
            query: str,
            documents: List[Document],
            **kwargs
    ) -> str:
        """
        Build a prompt from query and documents.

        Args:
            query: User's question
            documents: List of documents with citations (from prepare_citations())
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string ready for LLM
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            logger.warning("No documents provided, building prompt with empty context")

        logger.info(f"Building prompt for query: '{query[:50]}...' with {len(documents)} documents")

        try:
            # Build prompt using Haystack
            result = self.haystack_builder.run(
                query=query,
                documents=documents,
                **kwargs
            )

            prompt = result["prompt"]

            logger.info(f"Prompt built successfully ({len(prompt)} chars)")

            return prompt

        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            raise

    def build_prompt_with_citation_map(
            self,
            query: str,
            documents: List[Document],
            citation_map: Dict[str, Any],
            **kwargs
    ) -> Dict[str, Any]:
        """
        Build prompt and return with citation map for reference.

        Args:
            query: User's question
            documents: List of documents with citations
            citation_map: Citation map from prepare_citations()
            **kwargs: Additional template variables

        Returns:
            Dictionary with prompt and citation map
        """
        prompt = self.build_prompt(query, documents, **kwargs)

        return {
            "prompt": prompt,
            "citation_map": citation_map,
            "query": query,
            "num_documents": len(documents)
        }

    def get_template(self) -> str:
        """Get the current template string."""
        return self.template

    def set_template(self, template: str, required_variables: Optional[List[str]] = None):
        """
        Update the template.

        Args:
            template: New Jinja2 template string
            required_variables: Updated required variables list
        """
        self.template = template
        if required_variables:
            self.required_variables = required_variables

        # Reinitialize Haystack builder
        self.haystack_builder = HaystackPromptBuilder(
            template=self.template,
            required_variables=self.required_variables
        )

        logger.info("Template updated successfully")


def create_prompt_builder(template: Optional[str] = None) -> RAGPromptBuilder:
    """
    Factory function to create a RAGPromptBuilder instance.

    Args:
        template: Optional custom template (uses default if not provided)

    Returns:
        Configured RAGPromptBuilder instance
    """
    return RAGPromptBuilder(template=template)


# Additional template variants
CONCISE_TEMPLATE = """Answer the question using ONLY the provided context.

Context:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
{% endfor %}

Question: {{ query }}

Instructions:
- Cite sources using [1], [2], etc. after each claim
- Example: "The result is X [1]."
- Be brief and direct

Answer:"""

DETAILED_TEMPLATE = """You are a knowledgeable assistant. Provide a detailed answer based on the context.

Context Information:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
Source: {{ doc.meta.source }}{% if doc.meta.page %}, Page {{ doc.meta.page }}{% endif %}

{% endfor %}

User Question: {{ query }}

Instructions:
1. Base your answer STRICTLY on the provided context
2. Cite ALL sources using [1], [2], [3] etc. immediately after each relevant claim
3. Example: "Studies show X [1]. Further research confirms Y [2][3]."
4. Do NOT use [N] - use actual citation numbers
5. If information is missing, state: "The context does not provide information about..."
6. Provide comprehensive answer but stop after the answer - no additional notes

Your Answer:"""

STEP_BY_STEP_TEMPLATE = """You are a helpful assistant. Answer the question step by step using the context.

Context:
{% for doc in documents %}
[{{ loop.index }}] {{ doc.content }}
(Source: {{ doc.meta.source }})

{% endfor %}

Question: {{ query }}

Instructions:
- Break down your answer into clear steps
- Cite sources using [1], [2], etc. after each step
- Example: "Step 1: First, we observe X [1]."
- Only use information from the context
- Stop after your answer

Step-by-step Answer:"""


class PromptTemplates:
    """Collection of pre-defined prompt templates."""

    DEFAULT = DEFAULT_RAG_TEMPLATE
    CONCISE = CONCISE_TEMPLATE
    DETAILED = DETAILED_TEMPLATE
    STEP_BY_STEP = STEP_BY_STEP_TEMPLATE

    @staticmethod
    def get_template(name: str) -> str:
        """
        Get a template by name.

        Args:
            name: Template name ("default", "concise", "detailed", "step_by_step")

        Returns:
            Template string
        """
        templates = {
            "default": PromptTemplates.DEFAULT,
            "concise": PromptTemplates.CONCISE,
            "detailed": PromptTemplates.DETAILED,
            "step_by_step": PromptTemplates.STEP_BY_STEP
        }

        if name.lower() not in templates:
            logger.warning(f"Template '{name}' not found, using default")
            return PromptTemplates.DEFAULT

        return templates[name.lower()]
