"""
Clarification Request Handler for Conversational RAG.

This module handles clarification requests - queries where users express confusion
about the previous answer and need it revised for clarity.

Key Principles:
- Revise, don't expand
- Same length as original (±20%)
- Clearer language, same information
- NO new sources or retrieval
- Work only from previous answer

Examples:
- "I don't understand" / "I dont understand"
- "I'm confused" / "Im confused"
- "Can you clarify that?"
- "Explain this again"
"""

import re
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

# Subtype patterns for clarification requests (matched in order; first match wins).
# Used to select a dedicated prompt for:
# - "I don't understand" style confusion
# - "I don't get it" style confusion
# - "Explain this again" style explanation requests
CLARIFICATION_SUBTYPE_PATTERNS = {
    "dont_understand": [
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+understand(\s+it)?(\s+please)?\b",
        r"\b(please\s+)?(i\s+)?(don'?t|do\s+not|dont)\s+understand(\s+it)?\b",
        r"\b(i\s+)?(didn'?t|did\s+not|didnt)\s+understand(\s+it)?(\s+please)?\b",
        r"\b(please\s+)?(i\s+)?(didn'?t|did\s+not|didnt)\s+understand(\s+it)?\b",
        r"\b(can'?t|cannot|can\s+not|cant)\s+understand(\s+please)?\b",
        r"\b(i\s+)?(am|i'?m)\s+having\s+trouble\s+understanding(\s+it)?(\s+please)?\b",
        r"\b(i\s+)?(am|i'?m)\s+struggling\s+to\s+understand(\s+it)?(\s+please)?\b",
        r"\b(i\s+)?still\s+don'?t\s+understand(\s+it)?(\s+please)?\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+follow(\s+please)?\b",
        r"\b(i\s+)?(didn'?t|did\s+not|didnt)\s+follow(\s+please)?\b",
        r"\b(please\s+)?(i'?m|i\s+am|im)\s+confused\b",
        r"\b(i'?m|i\s+am|im)\s+confused\s+please\b",
        r"\b(i'?m|i\s+am|im)\s+still\s+confused(\s+please)?\b",
        r"\bthis\s+is\s+confusing(\s+me)?(\s+please)?\b",
        r"\bthat\s+was\s+confusing(\s+please)?\b",
        r"\b(i'?m|i\s+am|im)\s+not\s+following(\s+please)?\b",
        r"^confused(\s+please)?$",
        r"\b(not\s+)?(clear|unclear)(\s+please)?\b",
        r"\bhard\s+to\s+understand(\s+please)?\b",
        r"\btoo\s+complicated(\s+please)?\b",
        r"\beasier\s+to\s+understand(\s+please)?\b",
        r"\b(please\s+)?what\s+do\s+you\s+mean\b",
        r"\b(please\s+)?what\s+does\s+(that|this|it)\s+mean\b",
        r"\b(i\s+)?(am|i'?m)\s+not\s+sure\s+what\s+you\s+mean\b",
        r"\b(i\s+)?(am|i'?m)\s+not\s+sure\s+i\s+understand\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+quite\s+follow(\s+it|this|that)?\b",
        r"\b(i\s+)?(am|i'?m)\s+not\s+sure\s+i\s+follow\b",
        r"\b(i\s+)?(am|i'?m)\s+lost\b",
        r"\b(i\s+)?(am|i'?m)\s+missing\s+something\b",
        r"\b(that|this|it)\s+(didn'?t|did\s+not|didnt)\s+make\s+sense\b",
        r"\b(please\s+)?(can|could)\s+you\s+rephrase\b",
        r"\brephrase\s+(that|this|it)\b",
        r"\b(please\s+)?(can|could)\s+you\s+(put|say)\s+(that|this|it)\s+(differently|another\s+way)\b",
        r"\b(say|put)\s+(that|this|it)\s+another\s+way\b",
        r"\b(say|put)\s+(that|this|it)\s+differently\b",
        r"\b(can|could)\s+you\s+clarify\s+(that|this|it)\s+for\s+me\b",
        r"\bwhat\s+do\s+you\s+mean\s+by\s+(that|this|it)\b",
    ],
    "dont_get_it": [
        r"\b(please\s+)?(i\s+)?(don'?t|do\s+not|dont)\s+get\s+it\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+get\s+it\s+please\b",
        r"\b(please\s+)?(i\s+)?(didn'?t|did\s+not|didnt)\s+get\s+it\b",
        r"\b(i\s+)?(didn'?t|did\s+not|didnt)\s+get\s+(that|this)(\s+please)?\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+get\s+(that|this)(\s+please)?\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+really\s+get\s+(it|this|that)\b",
        r"\b(i\s+)?(am|i'?m)\s+not\s+getting\s+(it|this|that)\b",
        r"\b(doesn'?t|does\s+not|doesnt)\s+make\s+sense(\s+please)?\b",
        r"\bmakes?\s+no\s+sense(\s+please)?\b",
        r"\b(lost|losing)\s+(me|track)(\s+please)?\b",
        r"\bover\s+my\s+head(\s+please)?\b",
        r"\bnot\s+following(\s+please)?\b",
        r"\b(please\s+)?break\s+(it|that|this)\s+down\b",
        r"\b(i\s+)?(just\s+)?don'?t\s+get\s+this\b",
        r"\b(i\s+)?(just\s+)?don'?t\s+get\s+that\b",
    ],
    "explain": [
        # Direct "explain this/that/it" requests (with optional "please")
        r"\b(please\s+)?explain\s+(this|that|it)\b",
        r"\bexplain\s+(this|that|it)(\s+again)?\s*please\b",
        r"\b(please\s+)?explain\s+(this|that|it)\s+again\b",
        r"\b(please\s+)?explain\s+again\b",
        r"\bexplain\s+again\s+please\b",
        r"\b(please\s+)?explain\s+(better|differently)\b",
        r"\b(please\s+)?(can|could)\s+you\s+explain\b",
        r"^(please\s+)?explain\s+this\b",
        r"^(please\s+)?explain\s+that\b",
        r"^explain\s+(this|that)\s+please\b",
        r"\bexplain\s+it\s+to\s+me(\s+please)?\b",
        r"\bexplain\s+in\s+simple(r)?\s+(terms|language|words)\b",
        r"\bexplain\s+like\s+i('?m)?\s+\d+\b",
        r"^(please\s+)?explain\s+more(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?explain\s+further(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?explain\s+more\s+(about\s+)?(this|that|it)(\s+please)?[\.\!\?]*$",
        r"^(can|could)\s+you\s+explain\s+more(\s+please)?[\.\!\?]*$",
    ],
}

# Compiled patterns for each subtype (built once at module load).
_CLARIFICATION_SUBTYPE_COMPILED = {
    subtype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for subtype, patterns in CLARIFICATION_SUBTYPE_PATTERNS.items()
}


def classify_clarification_request(query: str) -> str:
    """
    Classify a clarification request into a subtype for prompt selection.

    Returns "dont_understand", "dont_get_it", "explain", or "generic".
    "dont_get_it" is checked before "dont_understand" so that
    "I don't get it" is not matched as "dont_understand".
    """
    if not query or not query.strip():
        logger.debug("Empty clarification query classified as generic")
        return "generic"
    q = query.strip()
    # Normalize apostrophes so "dont" / "don't" both match
    q_normalized = q.replace("'", "")
    # Check subtypes in order of specificity
    for subtype in ["dont_get_it", "dont_understand", "explain"]:
        for pattern in _CLARIFICATION_SUBTYPE_COMPILED[subtype]:
            if pattern.search(q) or pattern.search(q_normalized):
                logger.info("Clarification request classified as %s: %r", subtype, query[:80])
                return subtype

    logger.debug("Clarification request classified as generic: %r", query[:80])
    return "generic"


class ClarificationHandler:
    """
    Handler for clarification requests that revises previous answers.

    This handler is triggered when users express confusion about a previous answer.
    Instead of generating new content or retrieving new sources, it revises the
    existing answer to be clearer and easier to understand.

    Key Features:
    - Length-aware revision (targets same length ±20%)
    - No new information added
    - No source retrieval or citations
    - Clearer language without expansion
    - Structure preservation

    Example:
        ```python
        handler = ClarificationHandler(llm_generator)

        # User says "I don't understand" after getting an answer
        result = await handler.handle(
            query="I don't understand",
            last_turn=previous_turn,
            llm_generator=generator
        )

        # Result contains revised answer with no sources
        assert result["citations"] == []
        assert result["metadata"]["documents_used"] == 0
        ```
    """

    def __init__(self, llm_generator):
        """
        Initialize the clarification request handler.

        Args:
            llm_generator: LLM generator instance for generating revisions
        """
        self.llm_generator = llm_generator
        logger.info("ClarificationHandler initialized")

    def handle(
            self,
            query: str,
            last_turn: Any,
            llm_generator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Handle a clarification request query.

        This method revises the previous answer to be clearer without adding
        new information or retrieving new sources.

        Args:
            query: The clarification request (e.g., "I don't understand")
            last_turn: The previous ConversationTurn to revise
            llm_generator: Optional LLM generator (uses self.llm_generator if None)

        Returns:
            Dict containing:
                - answer: Revised answer text
                - citations: Empty list (no sources for revisions)
                - metadata: Dict with query_type, documents_used=0, revision info
        """
        generator = llm_generator or self.llm_generator

        # Validate last turn exists
        if not last_turn:
            logger.warning("No previous turn to revise")
            return {
                "answer": "There is no previous answer to clarify.",
                "citations": [],
                "metadata": {
                    "query_type": "clarification",
                    "documents_used": 0,
                    "is_revision": False,
                    "error": "no_previous_turn"
                }
            }

        # Validate last turn has an answer
        if not hasattr(last_turn, 'answer') or not last_turn.answer:
            logger.warning("Previous turn has no answer to revise")
            return {
                "answer": "There is no previous answer to clarify.",
                "citations": [],
                "metadata": {
                    "query_type": "clarification",
                    "documents_used": 0,
                    "is_revision": False,
                    "error": "no_previous_answer"
                }
            }

        # Use full answer when available (memory truncates to ~500 chars; clarification needs full text)
        original_answer_text = getattr(last_turn, "full_answer", None) or last_turn.answer
        if getattr(last_turn, "full_answer", None):
            logger.info(
                "Using full_answer for revision (len=%d); stored answer len=%d",
                len(original_answer_text), len(last_turn.answer)
            )
        else:
            logger.debug(
                "No full_answer on last turn; using stored answer (len=%d)",
                len(original_answer_text)
            )

        # Calculate target length (same as original ±20%)
        original_length = len(original_answer_text)
        target_min = int(original_length * 0.8)
        target_max = int(original_length * 1.2)
        target_length = (target_min + target_max) // 2

        logger.info(
            "Revising previous answer (original: %d chars, target: %d chars)",
            original_length, target_length,
        )
        # Log preview only (full text is in the prompt); use %s to avoid logging format errors
        logger.debug(
            "Full answer given to LLM prompt (preview): %.200s...",
            original_answer_text,
        )

        # Classify clarification subtype for dedicated prompt (dont_understand / dont_get_it / generic)
        clarification_subtype = classify_clarification_request(query)
        logger.debug("Clarification subtype for prompt selection: %s", clarification_subtype)

        # Build revision prompt (NO retrieval context); use full answer text and subtype-specific prompt
        prompt = self._build_revision_prompt(
            original_query=last_turn.query,
            original_answer=original_answer_text,
            clarification_request=query,
            target_length=target_length,
            original_length=original_length,
            clarification_subtype=clarification_subtype,
        )

        # Generate with length constraints and low temperature for focused output
        max_tokens = self._calculate_max_tokens(target_max)
        logger.info(f"Generating revision with max_new_tokens={max_tokens}, temperature=0.4")

        # Use low temperature for more focused, deterministic revisions
        revised = generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.4,
            purpose="clarification_rewrite"
        )

        # CRITICAL: Strict post-processing - use LLMGenerator's full pipeline (artifacts, meta-commentary, etc.)
        revised = generator._post_process(revised)
        logger.debug("Applied LLMGenerator _post_process for strict clarification output cleanup")

        # Clarification-specific meta-commentary strip (rewriting preambles, style instructions, etc.)
        revised = self._strip_meta_commentary(revised)

        revised_length = len(revised)
        length_ratio = revised_length / original_length if original_length > 0 else 0

        logger.info(f"Revision complete (length: {revised_length} chars, ratio: {length_ratio:.2f}x)")

        # Log warning if revision is too long
        if length_ratio > 1.5:
            logger.warning(f"Revision is {length_ratio:.2f}x longer than original - may need tuning")

        # Return response WITHOUT sources/citations
        return {
            "answer": revised,
            "citations": [],  # CRITICAL: No sources for revisions
            "metadata": {
                "query_type": "clarification",
                "clarification_subtype": clarification_subtype,
                "documents_used": 0,  # No new retrieval
                "is_revision": True,
                "original_length": original_length,
                "revised_length": revised_length,
                "length_ratio": length_ratio,
                "target_length": target_length
            }
        }

    @staticmethod
    def _get_explanation_instruction(clarification_request: str) -> str:
        """
        Return instruction block for explanation-style clarification requests,
        such as "Explain this" / "Explain that again" / "Can you explain it?".
        """
        OUTPUT_GUARDRAIL = (
            "Do not include any intro phrase (e.g., 'Here is the explanation', 'Sure!') "
            "or conversational filler — begin directly with the content."
        )

        return f'''{OUTPUT_GUARDRAIL}
The user asked you to EXPLAIN the previous answer more thoroughly, with a phrase like:
- "Explain this" / "Please explain this"
- "Explain that" / "Please explain that"
- "Explain it" / "Explain it please"
- "Explain this again" / "Please explain this again"
- "Can you explain this?" / "Please explain"

Their exact wording was: "{clarification_request}".

GOAL: Provide a thorough, in-depth explanation that genuinely helps the reader understand the topic fully.
- Elaborate on the key points — go deeper than the original answer, not just rephrase it.
- Expand on the reasoning, causes, implications, or context behind the facts.
- Add helpful background or clarifying detail that makes the topic easier to grasp.
- Stay focused on the same subject as the original answer; do not introduce unrelated topics.
- Do NOT cite new sources or documents — elaborate using explanation and context only.

METHOD:
- Open with a 1–2 sentence summary so the reader immediately knows what the topic is about.
- Work through each major idea in a logical order, giving each one enough depth to be understood.
- Define or unpack any terms, roles, or concepts that may not be obvious.
- Use concrete examples or comparisons to make abstract points tangible.
- Explain the significance or consequences of the key facts — why they matter.
- Use clear headers, paragraph breaks, and bullet points to organise the information.
- The explanation should be noticeably more detailed and thorough than the original answer.'''

    @staticmethod
    def _get_subtype_instruction(clarification_subtype: str, clarification_request: str) -> str:
        """
        Return instruction block for clarification-style requests.

        Args:
            clarification_subtype: "dont_understand", "dont_get_it", "explain", or "generic"
            clarification_request: The user's exact clarification phrase

        Returns:
            Instruction text to embed in the revision prompt
        """
        OUTPUT_GUARDRAIL = (
            "STRICT CONSTRAINT: Provide ONLY the rewritten content. "
            "Do not include any intro (e.g., 'Here is the rewrite') or conversational filler."
        )

        if clarification_subtype == "explain":
            return ClarificationHandler._get_explanation_instruction(clarification_request)

        return f'''{OUTPUT_GUARDRAIL}
The user indicated that they did not understand the previous answer with a phrase like:
- "I don't understand" / "Please, I don't understand" / "I don't understand please"
- "I dont get it" / "Please I dont get it"
- "I'm confused" / "I'm not following"
- "Can you explain that again?" / "Please explain"

Their exact wording was: "{clarification_request}".

GOAL: Rewrite the answer so that a reader who did not understand the first version can now clearly understand it.
- Keep all of the same facts and conclusions.
- Do NOT add new information or new sources.

METHOD:
- Use simple English where possible.
- Replace legal or technical terms with common words; if a term must remain, briefly define it in plain language.
- Break long sentences into shorter ones.
- Use clear paragraph breaks and bullet points to separate ideas.
- Make the main takeaway obvious in the first few sentences.'''

    @staticmethod
    def _build_revision_prompt(
            original_query: str,
            original_answer: str,
            clarification_request: str,
            target_length: int,
            original_length: int,
            clarification_subtype: str = "generic",
    ) -> str:
        """
        Build a prompt for revising the previous answer.

        Uses a subtype-specific instruction block when the clarification request
        matches "dont_understand" or "dont_get_it"; otherwise uses the generic block.

        Args:
            original_query: The user's original question
            original_answer: The previous answer to revise
            clarification_request: The clarification request (e.g., "I don't understand")
            target_length: Target character length for revision
            original_length: Original answer character length
            clarification_subtype: "dont_understand", "dont_get_it", or "generic"

        Returns:
            Formatted prompt string
        """
        subtype_instruction = ClarificationHandler._get_subtype_instruction(
            clarification_subtype, clarification_request
        )
        logger.debug(
            "Using revision prompt for subtype=%s, request_preview=%r",
            clarification_subtype,
            clarification_request[:60],
        )

        # Strict prompt: output only the revised answer; subtype instruction already includes user's request
        prompt = f'''{subtype_instruction}

Original answer to rewrite:
---
{original_answer}
---

STRICT RULES (violations are not allowed):
- Output ONLY the rewritten answer text. Nothing else.
- Do NOT add: preambles, "Here's...", "Yes - here it is...", "Sure!", "Let me...", or any phrase addressing the user.
- Do NOT mention: line count, "in fewer lines", "simple language", "as requested", or any meta-description of your output.
- Do NOT output any instruction or rule (e.g. "Do not use markdown", "use plain English", "write in paragraphs"). Output only the revised answer content.
- Do NOT invent or address imaginary constraints (e.g. "No more than 10 lines?" or similar). The user did not specify line limits.
- Do NOT include a second version or "alternative" after the first. Output exactly one clear revision.
- {"Elaborate and expand on the topic with more depth, context, and detail — do not simply rephrase the original." if clarification_subtype in ("explain", "dont_understand", "dont_get_it") else "Same facts only; shorter sentences; simpler words where helpful; no new information."}
- Do NOT output a refusal or "out of scope" message. You must revise the given answer only. Do not say "This question does not fall within that area" or "I focus on X, ask about Y instead"—rewrite the actual answer content for clarity.

### Anti-Meta-Commentary Rules (CRITICAL):
- Do NOT add internal reasoning, self-verification, or meta-commentary about your output.
- Do NOT include: "End of message", "This concludes", "No additional input required", "let me know how else", "within the defined parameters".
- Do NOT output: "Wait -", "correction needed", "Final Corrected Output", "Final Output:", "Proceeding with", "Confirmed operational", "System integrity", "Waiting for next".
- Do NOT add "Note:" lines that describe your output rather than the revised content.
- Output ONLY the direct revised answer. No self-referential commentary before or after.
- Do NOT add vague philosophical or moralising closing paragraphs. Do NOT end with sentences like "What matters most?", "But remember...", "every claim deserves careful review", "one thing stays constant", "perceptions shaped globally", or any reflective/rhetorical sign-off that adds no factual information. End on the last substantive fact or conclusion.

### Formatting Guidelines (CRITICAL - Follow These for Professional Structure):

**Structure your responses with clear organization:**

1. **Use descriptive headers** (use ## for main sections):
   - Start complex answers with a brief overview sentence
   - Use headers like "Legal Background", "Key Requirements", "Why This Matters", etc.
   - Headers should be clear and informative

2. **Use bullet points for lists and key information**:
   - Use * for main bullet points
   - Use - for sub-bullets when showing hierarchy
   - **CRITICAL: Indent sub-bullets with 3 spaces** to show they belong to the parent bullet
   - Group related points under headers
   - Keep bullets concise but complete

   **Correct nested list formatting:**
   * **Parent item**: Description
      - Sub-item one (note: 3 spaces before the dash)
      - Sub-item two (note: 3 spaces before the dash)
   * **Another parent**: Description
      - Nested detail
      - Another nested detail

3. **Use bold text for emphasis** (use **text** for bold):
   - Bold key terms, dates, or important names on first mention
   - Bold section outcomes or conclusions
   - Don't overuse - only for truly important information

4. **Structure multi-part answers clearly**:
   - Lead with a brief 1-2 sentence summary when appropriate
   - Break complex topics into logical sections with headers
   - Use white space (blank lines) between sections
   - Present information in order of importance or chronology

5. **For legal concepts, use this pattern**:
   Brief intro sentence.

   ## Main Category Header

   * **Key point with emphasis**: Explanation or details
   * **Another key point**: More details
      - Sub-point if needed (3 spaces before dash)
      - Additional sub-point (3 spaces before dash)

   ## Another Category

   [Continue pattern]

6. **Examples of good structure**:
   - Overview → Key Details (bulleted) → Related Information (bulleted) → Conclusion
   - Question restatement → Direct answer → Supporting details with headers → Related considerations

**Formatting DON'Ts:**
- Do not use emojis or emoticons
- Do not use em dashes (—)
- Never use citation markers like [1], [2], [3]
- Never mention whether you used the knowledge base or general knowledge
- Never include source filenames, page numbers, or references in the answer itself
- Do not use numbered lists unless explicitly requested or when showing a sequence of steps
- Do not use decorative formatting or ASCII art

Begin immediately with the first word of the revised answer—no greeting, no preamble, no phrase like "Here's", "Sure", or "Rewritten Answer:".'''

        logger.info(
            "Clarification revision prompt built: original_len=%d, target_len=%d, request=%r",
            original_length, target_length, clarification_request[:80],
        )
        logger.debug(
            "Clarification original_answer preview: %.200r...",
            original_answer[:200] if len(original_answer) > 200 else original_answer,
        )
        return prompt

    @staticmethod
    def _calculate_max_tokens(target_max_chars: int) -> int:
        """
        Calculate max_new_tokens based on target character length.

        Uses rough estimate of 1 token ≈ 4 characters.
        Adds 10% buffer to avoid premature truncation.

        Args:
            target_max_chars: Maximum target character length

        Returns:
            Maximum number of tokens to generate
        """
        # Rough estimate: 1 token ≈ 4 characters
        estimated_tokens = target_max_chars // 4

        # Add 25% buffer to avoid premature truncation (was 10%, raised to handle
        # cases where the stored answer was truncated by memory and the true original
        # is longer, so the rewrite legitimately needs more tokens).
        max_tokens = int(estimated_tokens * 1.25)

        # Ensure minimum of 512 tokens so even short originals produce a complete rewrite.
        # The previous floor of 100 caused responses to be cut mid-sentence when the
        # input was the memory-truncated version of a longer answer.
        max_tokens = max(512, max_tokens)

        # Cap at reasonable maximum (prevent runaway generation)
        max_tokens = min(max_tokens, 2048)

        return max_tokens

    def can_revise(self, last_turn: Any) -> bool:
        """
        Check if the last turn can be revised.

        Args:
            last_turn: The previous ConversationTurn

        Returns:
            True if revision is possible, False otherwise
        """
        if not last_turn:
            return False

        if not hasattr(last_turn, 'answer') or not last_turn.answer:
            return False

        # Don't revise if the last turn was itself a clarification
        if hasattr(last_turn, 'query_type') and last_turn.query_type == "clarification":
            logger.warning("Last turn was already a clarification, skipping nested revision")
            return False

        return True

    @staticmethod
    def _strip_meta_commentary(text: str) -> str:
        """
        Strip meta-commentary from LLM output.

        The LLM sometimes adds self-referential comments about its own output.
        This method removes them to keep only the actual revised content.

        Args:
            text: The raw LLM output

        Returns:
            Cleaned text without meta-commentary
        """
        # Patterns that indicate meta-commentary (case insensitive)
        meta_patterns = [
            r'\n\n(Make it|Should I|Want me to|Would you like|Do you want).*$',
            r'\n\n\*\*Short(er)? version:?\*\*.*$',
            r'\n\nThis (version|rewrite|revision).*$',
            r'\n\nNote:.*$',
            r'\n\nI\'ve.*$',
            r'\n\nThe above.*$',
            r'\n\nNow I.*$',
            r'\n\n---\s*$',
            r'\n\nHere\'s.*$',
            r'\n\nLet me.*$',
            # Word count or formatting notes
            r'\(Word count[^)]*\).*$',
            r'\(.*reduced from[^)]*\).*$',
            r' - concise while keeping.*$',
            # Nested rewrite instructions / conversational filler (remove to end)
            r'\n\nNow rewrite again.*$',
            r'\n\nSure!.*$',
            r"\n\nHere's a friendlier.*$",
            r'\n\n(same meaning|shorter and clearer).*$',
            r'\n\n.*conversational version.*$',
            # Imaginary constraints and "here it is" commentary
            r'^.*?[Nn]o more than \d+ lines\?.*?\n',
            r'\n\n[Yy]es\s*[-—]\s*here it is[^.\n]*(in fewer lines|with simple language)[^.]*[.:]\s*\n',
            r'\n\n[Yy]es\s*[-—].*?here (it is|you go)[^.]*[.:]\s*\n',
            r'\n\n.*?in fewer lines with simple language[.:]?\s*\n',
            # Trailing model-generated follow-up question (from conversation JSONs)
            r'\n\nWhat happens if I don\'t follow up\?[^\n]*\n.*$',
            r'\n\nCan we just keep doing things informally\?.*$',
            # Trailing heading that announces the rewrite / final version
            # e.g. "# Rewritten Version Based On Your Request"
            r'\n\n#+\s*(Rewritten|Revised|Final|Updated|Clean)\s+(Version|Answer|Response|Output)[^\n]*.*$',
            # Trailing parenthetical compliance notes
            # e.g. "(Only final clean response below - follows ALL rules strictly.)"
            r'\n\n?\(Only (final|clean|corrected)[^)]*\)[\s.]*$',
            r'\(Only (final|clean|corrected)[^)]*follows ALL rules[^)]*\)[\s.]*$',
            # "All content derived from prior response" compliance notes (inline or trailing)
            r'\n?\(All content derived[^)]*\)[\s.]*$',
            r'\n?\(Content derived[^)]*\)[\s.]*$',
            r'\n?\(Derived (?:strictly )?from[^)]*\)[\s.]*$',
            r'\n?\((?:Based|Sourced) (?:strictly )?(?:on|from) (?:prior|previous|the original)[^)]*\)[\s.]*$',
            # Lone markdown heading used as separator after content (e.g. "#\n" or "##\n")
            r'\n\n#+\s*\n\(.*?\)[\s.]*$',
            r'\n\n#+\s*$',
            # "follows ALL rules strictly" and similar compliance sign-offs anywhere at the end
            r'\n\n?\(.*?follows ALL rules strictly\.?\)[\s.]*$',
            r'\n\n?\(.*?strictly follows all rules\.?\)[\s.]*$',
            # Conversational self-correction / task-level meta-commentary
            # e.g. "You're right - we need clearer separation..."
            r"\n\nYou'?re right\b.*$",
            r'\n\nWe\'?re still missing.*$',
            r'\n\nWe need (clearer|better|more).*$',
            # Blockquote lines referencing the original prompt / task scope
            # e.g. "> The original prompt asked us to rephrase just ONE specific part..."
            r'\n\n> (?:The|This) original\b.*$',
            r'\n\n> .*\b(?:original prompt|asked us to|rephrase just)\b.*$',
            # Generic trailing blockquote at the very end (model using > as a note/aside)
            r'\n\n>(?:[^\n]*\n)*[^\n]*$',
            # Self-referential "we/I" commentary about what was/wasn't done
            r'\n\nWe\'?re (?:still|also) (?:missing|ignoring|skipping|omitting).*$',
            r'\n\nI (?:missed|forgot|skipped|omitted|left out).*$',
        ]

        # Mid-response meta-reasoning blocks (e.g. "### Response Strategy:")
        # These appear between content sections and must be removed without discarding
        # the legitimate content that follows. Applied separately with re.sub so the
        # content after the block is preserved.
        _META_BLOCK_RE = re.compile(
            r'\n\n'
            r'###\s+(?:Response\s+)?(?:Strategy|Plan|Approach|Reasoning|Note)s?:?[^\n]*'
            r'(?:\n(?!##)[^\n]*)*'
            r'\n*',
            re.IGNORECASE,
        )
        # Strip leading commentary (e.g. "No more than 10 lines? Yes - here it is in fewer lines with simple language:")
        # Order: most specific first (from data/conversations/*.json and query_responses.json turn-2 scan)
        leading_commentary = [
            # --- From conversation JSONs: "Just start rewriting directly from the top" + follow-up line
            r'^Just start rewriting directly from the top\.\s*\n+',
            r'^We are told we cannot write anything other than the rewritten answer[^\n]*\n+',
            # --- "Just start rewriting directly" + "We are going to rewrite the original answer..."
            r'^Just start rewriting directly\.\s*\n+',
            r'^We are going to rewrite the original answer[^\n]*\n+',
            # session_b4bbf61b35f5.json: "We are rewriting the provided original answer..."
            r'^We are rewriting the provided original answer[^\n]*\n+',
            # --- "Just start at the beginning and end where the original ends" + instruction block
            r'^Just start at the beginning and end where the original ends\.\s*\n+',
            r'^If you see an incomplete sentence such as[^\n]+complete it logically[^\n]*\n+',
            r'^Use standard punctuation including periods, commas[^\n]*\n+',
            r'^Ensure all legal terms remain accurate[^\n]*\n+',
            r'^Final check: Is every part of the response[^\n]*\?\s*\n+',
            # --- Variant seen in live clarification output: "Just start directly with the corrected ..."
            r'^Just start directly with the corrected (text|response) using proper header format and structured layout\.\s*\n+',
            # --- Other "Just start..." / "Just write..." variants (session_c070720fdbdd.json, session_07feac0bf9fc.json)
            r'^Just start with the corrected text using proper structure as described above\.\s*\n+',
            r'^Just write the corrected response using proper structure and format\.\s*\n+',
            r'^Just start writing the corrected and clarified response as if this were always its final form\.\s*\n+',
            r'^Just start writing the corrected and clarified response\.\s*\n+',
            r'^Just start writing the corrected response as if you were directly providing an improved explanation\.\s*\n+',
            r'^Just start writing the corrected sentence\(s\)\.\s*\n+',
            r'^Just start at the beginning\.\s*\n+',
            r'^Just start writing\.\s*\n+',
            # --- Legacy / scan doc patterns
            r'^[^\n]*(?:no more than \d+ lines\?|yes\s*[-—]\s*here it is)[^\n]*(?:\n|$)',
            r'^[^\n]*in fewer lines with simple language[.:]?\s*\n+',
            r'^No (intro|outro) line\.?\s*\n*',
            # Instruction-echo: "Do not use markdown...", "just plain English...", etc.
            r'^(?:Do not use|Don\'?t use|Avoid|Use only)[^\n]*(?:markdown|formatting|plain English)[^\n]*\n+',
            r'^[^\n]*(?:just |only )?plain English[^\n]*\n+',
            # LLM echo of style instructions (turn2_clarification_scan.md: Gillmore, Step-Parent, FC 271, Professional Goodwill)
            r'^Use standard punctuation(?: and capitalization)?(?: as appropriate)?(?: for natural reading flow)?\.?\s*\n*',
            r'^Use standard punctuation(?: including periods, commas[^\n]*)?\.?\s*\n*',
            r'^Use standard punctuation\.\s*\n*',
            r'^Use active voice where possible\.\s*Avoid passive constructions unless necessary\.?\s*\n*',
            r'^No markdown formatting at all\.?\s*\n*',
            r'^Rewritten [Aa]nswer:?\s*\n*',
            r'^Just start writing the corrected response[^\n]*[.!?]?\s*(?:\n|$)',
            r'^Just start (at the beginning|writing)[^\n]*(?:\n|$)',
        ]

        cleaned = text
        cleaned = cleaned.strip()
        cleaned = re.sub(r'^-{3,}\s*\n+', '', cleaned)
        cleaned = re.sub(r'^"{3}\s*\n*', '', cleaned)
        cleaned = re.sub(r'\n*"{3}\s*$', '', cleaned)

        # ── Fast first-line check ────────────────────────────────────────────
        # The LLM sometimes echoes a single meta-instruction line before the
        # real content, e.g. "Just start writing the corrected response using
        # proper header and bullet format as described above."
        # Regex newline-anchored patterns (\.\s*\n+) can silently fail when
        # \s* greedily consumes all newlines before \n+ can match.
        # A simple split-on-first-line approach is more reliable.
        _META_FIRST_LINE_PREFIXES = (
            "just start writing",
            "just start rewriting",
            "just start with",
            "just write the corrected",
            "just begin",
            "begin immediately with",
            "here is the rewritten",
            "here is the revised",
            "rewritten answer:",
            "revised answer:",
        )
        first_newline = cleaned.find('\n')
        if first_newline != -1:
            first_line = cleaned[:first_newline].strip().lower()
            if any(first_line.startswith(p) for p in _META_FIRST_LINE_PREFIXES):
                logger.info(
                    "Stripped meta-commentary first line (%d chars): %r",
                    first_newline, cleaned[:first_newline],
                )
                cleaned = cleaned[first_newline:].lstrip()

        # Repeatedly strip leading commentary so multi-line blocks are fully removed
        for _ in range(15):
            prev_len = len(cleaned)
            for pattern in leading_commentary:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            if len(cleaned) == prev_len:
                break
        # Strip "Rewritten Answer:" and style instructions that may appear after a newline
        cleaned = re.sub(r'\n+Rewritten [Aa]nswer:?\s*\n*', '\n\n', cleaned, count=1)
        # Strip leading lines that are only style instructions (multi-line leading block)
        # Includes patterns from turn2_clarification_scan.md and conversation JSONs
        instruction_line = (
            r'^(Use standard punctuation[^\n]*|Use active voice[^\n]*|Avoid passive constructions[^\n]*|'
            r'No markdown formatting[^\n]*|Rewritten [Aa]nswer:?|'
            r'Just (start (rewriting|writing|with the corrected text)|write the corrected response)[^\n]*|We are told we cannot write[^\n]*|'
            r'We are (going to rewrite the original answer|rewriting the provided original answer)[^\n]*|'
            r'If you see an incomplete sentence[^\n]*|Ensure all legal terms[^\n]*|Final check:[^\n]*)\s*\n'
        )
        for _ in range(15):  # allow multiple leading instruction lines
            line_stripped = re.sub(instruction_line, '', cleaned, count=1, flags=re.IGNORECASE)
            if line_stripped == cleaned:
                break
            cleaned = line_stripped
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Strip mid-response meta-reasoning blocks (e.g. "### Response Strategy:").
        # Uses a replace (not truncation) so legitimate content after the block is kept.
        if _META_BLOCK_RE.search(cleaned):
            before_len = len(cleaned)
            cleaned = _META_BLOCK_RE.sub('\n\n', cleaned).strip()
            logger.info(
                "Stripped mid-response meta-reasoning block from clarification output: %d -> %d chars",
                before_len, len(cleaned),
            )

        # Strip trailing heading-style separators followed by a parenthetical compliance note,
        # e.g. "\n# Rewritten Version...\n#\n(Only final clean response below...)"
        # Covers any variation the model generates after the actual answer content.
        cleaned = re.sub(
            r'\n\n#+[^\n]*\n+#+\s*\n\([^\)]*\)\s*$',
            '',
            cleaned,
            flags=re.DOTALL,
        )
        # Strip any trailing line that is purely a markdown heading acting as a separator
        cleaned = re.sub(r'\n\n+#+\s*[^\n]*\s*$', '', cleaned, flags=re.DOTALL)

        # Also strip trailing whitespace
        cleaned = cleaned.strip()

        # Log if we stripped anything
        if len(cleaned) < len(text):
            logger.info(
                "Stripped meta-commentary from clarification output: %d -> %d chars",
                len(text), len(cleaned)
            )
            logger.debug("Removed snippet (last 200 chars): %s", text[max(0, len(text) - 200):])

        return cleaned
