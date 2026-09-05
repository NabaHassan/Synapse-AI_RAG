"""
Formatting Request Handler for Conversational RAG.

This module handles formatting requests - queries asking to reformat
the previous answer without generating new content.

Examples:
- "with proper headings and bullets"
- "make it a list"
- "reformat that"
"""

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FormattingRequestHandler:
    """
    Handler for formatting request queries.
    
    Reformats the last answer without performing new retrieval.
    Reuses the same source documents from the previous turn.
    
    This saves resources and ensures consistency.
    """

    def __init__(self, llm_generator):
        """
        Initialize the formatting request handler.
        
        Args:
            llm_generator: LLM generator instance for reformatting
        """
        self.llm_generator = llm_generator
        logger.info("FormattingRequestHandler initialized")

    def handle(
            self,
            query: str,
            last_turn: Any,
            llm_generator: Optional[Any] = None
    ) -> str:
        """
        Handle a formatting request query.

        Args:
            query: The formatting request (e.g., "with bullets")
            last_turn: The previous ConversationTurn to reformat
            llm_generator: Optional LLM generator (uses self.llm_generator if None)

        Returns:
            Reformatted answer string
        """
        generator = llm_generator or self.llm_generator

        if not last_turn:
            logger.warning("FormattingRequestHandler.handle: no previous turn to reformat")
            return "There is no previous answer to reformat."

        # Prefer full_answer so the reformat is based on the complete prior response,
        # not the memory-truncated copy stored in last_turn.answer.
        original_answer = getattr(last_turn, "full_answer", None) or last_turn.answer
        if getattr(last_turn, "full_answer", None):
            logger.info(
                "Formatting: using full_answer (%d chars) instead of truncated stored answer (%d chars)",
                len(original_answer), len(last_turn.answer),
            )
        original_answer = self._strip_formatting_artifacts(original_answer)

        # Build reformatting prompt and detect whether this is a concise/summary request
        prompt, is_concise = self._build_reformatting_prompt(
            original_query=last_turn.query,
            original_answer=original_answer,
            format_request=query
        )

        # Use lower temperature for concise/summary requests so output is tight and focused
        temperature = 0.1 if is_concise else None
        logger.info(
            "Reformatting previous answer with request: '%s' | is_concise=%s | temperature=%s",
            query, is_concise, temperature,
        )
        reformatted = generator.generate(prompt, temperature=temperature, purpose="response_formatting")
        logger.debug("Raw formatting output length: %d chars", len(reformatted))

        # Strip any meta-commentary the LLM may have added
        reformatted = self._strip_meta_commentary(reformatted)
        logger.debug("Formatting output length after strip: %d chars", len(reformatted))

        return reformatted

    @staticmethod
    def _strip_meta_commentary(text: str) -> str:
        """
        Strip meta-commentary from LLM output so only the reformatted content is returned.

        Applies the same comprehensive pipeline used by ClarificationHandler:
        leading-commentary stripping, mid-response meta-block removal, trailing
        heading/parenthetical cleanup, and the full meta_patterns list.
        """
        if not text or not text.strip():
            return text
        cleaned = FormattingRequestHandler._strip_formatting_artifacts(text)

        logger.debug("Raw formatting output before strip (first 200 chars): %r", cleaned[:200])

        # ── Trailing / inline meta patterns ─────────────────────────────────
        meta_patterns = [
            # Formatting-handler-specific leading preambles
            r'^Here (?:is|are) (?:the |a )?reformatted (?:answer|version)[.:\s]*\n*',
            r'^(?:Sure!?|Okay\.?)\s*[:\s]*\n*',
            # Leading echo of the formatting request itself (e.g., "Make this shorter.")
            r'^(?:make|keep)\s+(?:it|this|that|the\s+answer)\s+(?:short|shorter|concise|brief)[\.\!\?]*\s*\n*',
            r'^make\s+this\s+shorter[\.\!\?]*\s*\n*',
            # Inline / trailing note lines
            r'\n\n(?:Note|I\'ve|The above|Here\'s|Let me)[^\n]*.*$',
            r'\n\n---\s*$',
            r'\(Word count[^)]*\).*$',
            r'\(.*reduced from[^)]*\).*$',
            r' - (?:concise|same meaning)[^\n]*$',
            r' - concise while keeping.*$',
            # Self-referential meta about violating instructions or adding content
            r'\n\n(?:Wait!?\s*I just realized|I cannot include anything beyond|That violates your instructions)[\s\S]*$',
            # Generic closing / sign-off commentary
            r'\n\nYes\.[\s\S]*?\(No additional action required\.\)[\s\S]*$',
            # Clarification-handler patterns ported here for full coverage
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
            # Trailing model-generated follow-up questions
            r'\n\nWhat happens if I don\'t follow up\?[^\n]*\n.*$',
            r'\n\nCan we just keep doing things informally\?.*$',
            # Trailing heading that announces the rewrite / final version
            r'\n\n#+\s*(Rewritten|Revised|Final|Updated|Clean)\s+(Version|Answer|Response|Output)[^\n]*.*$',
            # Trailing parenthetical compliance notes
            r'\n\n?\(Only (final|clean|corrected)[^)]*\)[\s.]*$',
            r'\(Only (final|clean|corrected)[^)]*follows ALL rules[^)]*\)[\s.]*$',
            # "All content derived from prior response" compliance notes (inline or trailing)
            r'\n?\(All content derived[^)]*\)[\s.]*$',
            r'\n?\(Content derived[^)]*\)[\s.]*$',
            r'\n?\(Derived (?:strictly )?from[^)]*\)[\s.]*$',
            r'\n?\((?:Based|Sourced) (?:strictly )?(?:on|from) (?:prior|previous|the original)[^)]*\)[\s.]*$',
            # Lone markdown heading used as separator after content
            r'\n\n#+\s*\n\(.*?\)[\s.]*$',
            r'\n\n#+\s*$',
            # "follows ALL rules strictly" compliance sign-offs
            r'\n\n?\(.*?follows ALL rules strictly\.?\)[\s.]*$',
            r'\n\n?\(.*?strictly follows all rules\.?\)[\s.]*$',
            # Conversational self-correction / task-level meta-commentary
            r"\n\nYou'?re right\b.*$",
            r'\n\nWe\'?re still missing.*$',
            r'\n\nWe need (clearer|better|more).*$',
            # Blockquote lines referencing the original prompt / task scope
            r'\n\n> (?:The|This) original\b.*$',
            r'\n\n> .*\b(?:original prompt|asked us to|rephrase just)\b.*$',
            # Generic trailing blockquote at the very end
            r'\n\n>(?:[^\n]*\n)*[^\n]*$',
            # Self-referential "we/I" commentary about what was/wasn't done
            r'\n\nWe\'?re (?:still|also) (?:missing|ignoring|skipping|omitting).*$',
            r'\n\nI (?:missed|forgot|skipped|omitted|left out).*$',
            # Prompt artifact leaking into output — strip this sentinel and everything after
            r'\n\nYou\'?re now ready to respond based on the above rules\..*$',
            r'\n\nYou are now ready to respond based on the above rules\..*$',
            # Internal LLM reasoning / chain-of-thought leaking into output
            r'\n\nHmm,?\s+I need to\b.*$',
            r'\n\nLooking at the (?:original|provided) (?:response|answer)\b.*$',
            r'\n\nI need to create\b.*$',
            r'\n\nI\'?m going to\b.*$',
            r'\n\nLet me (?:think|consider|analyze|review|re-read|look at)\b.*$',
        ]

        # ── Mid-response meta-reasoning blocks (e.g. "### Response Strategy:") ──
        _META_BLOCK_RE = re.compile(
            r'\n\n'
            r'###\s+(?:Response\s+)?(?:Strategy|Plan|Approach|Reasoning|Note)s?:?[^\n]*'
            r'(?:\n(?!##)[^\n]*)*'
            r'\n*',
            re.IGNORECASE,
        )

        # ── Leading commentary patterns ──────────────────────────────────────
        leading_commentary = [
            r'^Just start rewriting directly from the top\.\s*\n+',
            r'^We are told we cannot write anything other than the rewritten answer[^\n]*\n+',
            r'^Just start rewriting directly\.\s*\n+',
            r'^We are going to rewrite the original answer[^\n]*\n+',
            r'^We are rewriting the provided original answer[^\n]*\n+',
            r'^Just start at the beginning and end where the original ends\.\s*\n+',
            r'^If you see an incomplete sentence such as[^\n]+complete it logically[^\n]*\n+',
            r'^Use standard punctuation including periods, commas[^\n]*\n+',
            r'^Ensure all legal terms remain accurate[^\n]*\n+',
            r'^Final check: Is every part of the response[^\n]*\?\s*\n+',
            r'^Just start directly with the corrected (text|response) using proper header format and structured layout\.\s*\n+',
            r'^Just start with the corrected text using proper structure as described above\.\s*\n+',
            r'^Just write the corrected response using proper structure and format\.\s*\n+',
            r'^Just start writing the corrected and clarified response as if this were always its final form\.\s*\n+',
            r'^Just start writing the corrected and clarified response\.\s*\n+',
            r'^Just start writing the corrected response as if you were directly providing an improved explanation\.\s*\n+',
            r'^Just start writing the corrected sentence\(s\)\.\s*\n+',
            r'^Just start at the beginning\.\s*\n+',
            r'^Just start writing\.\s*\n+',
            r'^[^\n]*(?:no more than \d+ lines\?|yes\s*[-—]\s*here it is)[^\n]*(?:\n|$)',
            r'^[^\n]*in fewer lines with simple language[.:]?\s*\n+',
            r'^No (intro|outro) line\.?\s*\n*',
            r'^(?:Do not use|Don\'?t use|Avoid|Use only)[^\n]*(?:markdown|formatting|plain English)[^\n]*\n+',
            r'^[^\n]*(?:just |only )?plain English[^\n]*\n+',
            r'^Use standard punctuation(?: and capitalization)?(?: as appropriate)?(?: for natural reading flow)?\.?\s*\n*',
            r'^Use standard punctuation(?: including periods, commas[^\n]*)?\.?\s*\n*',
            r'^Use standard punctuation\.\s*\n*',
            r'^Use active voice where possible\.\s*Avoid passive constructions unless necessary\.?\s*\n*',
            r'^No markdown formatting at all\.?\s*\n*',
            r'^Rewritten [Aa]nswer:?\s*\n*',
            r'^Just start writing the corrected response[^\n]*[.!?]?\s*(?:\n|$)',
            r'^Just start (at the beginning|writing)[^\n]*(?:\n|$)',
        ]

        # ── Fast first-line check ────────────────────────────────────────────
        _META_FIRST_LINE_PREFIXES = (
            "just start writing",
            "just start rewriting",
            "just start with",
            "just write the corrected",
            "just begin",
            "begin immediately with",
            "here is the rewritten",
            "here is the revised",
            "here is the reformatted",
            "here is a concise",
            "here is a shorter",
            "here is a summary",
            "rewritten answer:",
            "revised answer:",
            "reformatted answer:",
            "condensed answer:",
            "summary:",
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

        # Strip "Rewritten Answer:" label if it appears after a newline
        cleaned = re.sub(r'\n+Rewritten [Aa]nswer:?\s*\n*', '\n\n', cleaned, count=1)

        # Strip leading lines that are purely style instructions
        instruction_line = (
            r'^(Use standard punctuation[^\n]*|Use active voice[^\n]*|Avoid passive constructions[^\n]*|'
            r'No markdown formatting[^\n]*|Rewritten [Aa]nswer:?|'
            r'Just (start (rewriting|writing|with the corrected text)|write the corrected response)[^\n]*|'
            r'We are told we cannot write[^\n]*|'
            r'We are (going to rewrite the original answer|rewriting the provided original answer)[^\n]*|'
            r'If you see an incomplete sentence[^\n]*|Ensure all legal terms[^\n]*|Final check:[^\n]*)\s*\n'
        )
        for _ in range(15):
            line_stripped = re.sub(instruction_line, '', cleaned, count=1, flags=re.IGNORECASE)
            if line_stripped == cleaned:
                break
            cleaned = line_stripped

        # Apply trailing / inline meta patterns
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

        # Strip mid-response meta-reasoning blocks
        if _META_BLOCK_RE.search(cleaned):
            before_len = len(cleaned)
            cleaned = _META_BLOCK_RE.sub('\n\n', cleaned).strip()
            logger.info(
                "Stripped mid-response meta-reasoning block from formatting output: %d -> %d chars",
                before_len, len(cleaned),
            )

        # Strip trailing heading-separator + parenthetical compliance note combos
        cleaned = re.sub(
            r'\n\n#+[^\n]*\n+#+\s*\n\([^\)]*\)\s*$',
            '',
            cleaned,
            flags=re.DOTALL,
        )
        cleaned = re.sub(r'\n\n+#+\s*[^\n]*\s*$', '', cleaned, flags=re.DOTALL)

        # Remove explicit commentary labels while preserving the actual content.
        # Example: "* **Shorter version**: If requesting ..." -> "* If requesting ..."
        cleaned = re.sub(
            r'^\*\s*\*\*shorter version\*\*:\s*',
            '* ',
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Drop purely decorative headings that describe the meta operation, not content.
        # Example: "## Key Points Summary"
        cleaned = re.sub(
            r'^##\s*key points summary\s*\n?',
            '',
            cleaned,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        cleaned = cleaned.strip()
        if len(cleaned) < len(text):
            logger.info(
                "Stripped meta-commentary from formatting output: %d -> %d chars",
                len(text), len(cleaned),
            )
            logger.debug("Removed snippet (last 200 chars): %s", text[max(0, len(text) - 200):])
        return cleaned

    @staticmethod
    def _strip_formatting_artifacts(text: str) -> str:
        """
        Remove common formatting artifacts that can leak into reformatting.
        """
        if not text or not text.strip():
            return text
        cleaned = text.strip()
        cleaned = re.sub(r'^-{3,}\s*\n+', '', cleaned)
        cleaned = re.sub(r'^"{3}\s*\n*', '', cleaned)
        cleaned = re.sub(r'\n*"{3}\s*$', '', cleaned)
        return cleaned.strip()

    @staticmethod
    def _get_concise_instruction(format_request: str) -> str:
        """
        Return instruction block for conciseness / summary-style formatting requests.

        Examples of user phrases this should cover:
        - "make it concise" / "please make it concise" / "make it concise please"
        - "make this shorter" / "please make it shorter"
        - "short version" / "shorter version"
        - "concise version" / "condense it" / "compress it"
        - "tl;dr"
        """
        return f'''The user asked for a more concise / summarized version of the previous answer with a request like:
- "make it concise" / "please make it concise" / "make it concise please"
- "make this shorter" / "please make it shorter"
- "short version" / "shorter version"
- "concise version" / "condense it" / "compress it"
- "tl;dr"

Their exact wording was: "{format_request}".

GOAL: Produce a tightly compressed version of the previous answer that preserves every important fact while removing all unnecessary words.
- Every sentence in the output MUST come directly from the original answer — no paraphrasing that changes meaning, no invented details.
- Remove only: repetition, filler phrases, verbose phrasing, and minor side details not central to the main point.
- Keep all core facts, names, dates, figures, and conclusions exactly as stated in the original.

METHOD:
- Start with a 1–2 sentence summary of the overall topic.
- Then list only the most critical points in tight bullet points or short paragraphs.
- Use simple, direct sentences — cut every word that does not add meaning.
- Preserve the logical order of the original answer.

STRICT OUTPUT RULES (violations are not allowed):
- Output ONLY the condensed answer. Nothing else.
- Do NOT add any preamble: no "Here is a concise version", no "Sure!", no "As requested", no greeting.
- Do NOT add any closing remark: no "Let me know if you'd like more", no "Hope that helps", no sign-off.
- Do NOT add meta-commentary about what you did: no "I removed X", no "I kept only Y", no "Note:", no "Word count:".
- Do NOT invent new facts, names, dates, statistics, or sources that were not in the original answer.
- Do NOT quote or reference the original prompt or these instructions in your output.
- Do NOT output any self-referential text such as "End of summary", "This concludes", "Final answer:", or similar.
- Begin immediately with the first word of the condensed answer.'''

    @staticmethod
    def _build_reformatting_prompt(
            original_query: str,
            original_answer: str,
            format_request: str
    ) -> tuple:
        """
        Build a prompt for reformatting the previous answer.

        Detects the type of formatting request and provides specific instructions.

        Returns:
            Tuple of (prompt: str, is_concise: bool) where is_concise indicates
            whether this is a summarise/condense/shorten request that should use
            a lower generation temperature.
        """
        request_lower = format_request.lower().replace('’', "'")
        is_concise = False

        # Detect type of formatting request and set appropriate instruction
        if any(
                phrase in request_lower
                for phrase in [
                    "more explanatory",
                    "more detailed",
                    "more clear",
                    "more specific",
                    "explain more",
                    "explain further",
                    "tell me more",
                    "more detail",
                    "more details",
                    "more information",
                    "more info",
                    "give me details",
                    "please give me details",
                    "can you give me details",
                    "could you please give me details",
                    "go deeper",
                    "go into more detail",
                    "expand on",
                    "elaborate on",
                    "more context",
                    "walk me through",
                    "unpack",
                    "drill down on",
                ]
        ):
            instruction = "Expand and elaborate on the previous answer with MORE DETAIL and EXPLANATION. Add examples, clarifications, and additional context where appropriate."

        elif "make it better" in request_lower or "improve" in request_lower or "enhance" in request_lower:
            instruction = "Improve the previous answer by making it CLEARER, MORE COMPREHENSIVE, and BETTER ORGANIZED. Fix any ambiguities and add helpful details."

        elif "expand" in request_lower or "elaborate" in request_lower:
            instruction = "EXPAND on the previous answer with additional information, examples, and explanations. Make it more thorough and complete."

        elif ("simplify" in request_lower or "simpler" in request_lower or "terms" in request_lower
              or "easier" in request_lower or "layman" in request_lower or "plain english" in request_lower):
            instruction = "SIMPLIFY the previous answer. Use simpler language, shorter sentences, and clearer explanations. Remove unnecessary complexity (e.g. layman's terms, plain English)."

        elif (
                "summarize" in request_lower
                or "summarise" in request_lower
                or "summary" in request_lower
                or "tl;dr" in request_lower
                or "tldr" in request_lower
                or "concise" in request_lower
                or "brief" in request_lower
                or "shorter" in request_lower
                or "short version" in request_lower
                or "shorter version" in request_lower
                or "condense" in request_lower
                or "compress" in request_lower
                or "gist" in request_lower
                or "nutshell" in request_lower
                or "boil down" in request_lower
                or "boil it down" in request_lower
                or "to the point" in request_lower
                or "get to the point" in request_lower
                or "main points" in request_lower
                or "make it short" in request_lower
                or "keep it short" in request_lower
                or "in short" in request_lower
        ):
            instruction = FormattingRequestHandler._get_concise_instruction(format_request)
            is_concise = True
            logger.debug("Concise/summary formatting request detected — temperature will be set to 0.1")

        elif "bullet" in request_lower or "list" in request_lower or "step by step" in request_lower:
            instruction = (
                "Reformat the previous answer using BULLET POINTS and CLEAR STRUCTURE.\n"
                "Use * for main bullet points. If you need sub-points, indent them with three spaces and use '-'.\n"
                "Group related sub-points under a parent bullet rather than listing every line as a top-level bullet."
            )

        elif "heading" in request_lower or "header" in request_lower:
            instruction = "Reformat the previous answer with PROPER HEADINGS and SECTIONS. Use markdown formatting with ## for main sections and ### for subsections."

        elif "rewrite" in request_lower or "rephrase" in request_lower:
            instruction = "REWRITE the previous answer using different words and sentence structures while maintaining the same meaning and information."

        else:
            instruction = "Reformat the previous answer according to the user's request."

        prompt = f'''The user previously asked: "{original_query}"

The answer was:
"""
{original_answer}
"""

Now the user wants you to reformat this answer. Their request is: "{format_request}"

{instruction}

CRITICAL INSTRUCTIONS:
- Do NOT retrieve new information or add facts not in the original answer
- Use ONLY the information from the previous answer
- Maintain accuracy - don't change the meaning
- Format and present the existing information as requested
- Output ONLY the reformatted answer: no preamble, no "Here is...", no meta-commentary, no notes about what you did

Reformatted answer:'''

        return prompt, is_concise

    def can_reformat(self, last_turn: Any) -> bool:
        """
        Check if the last turn can be reformatted.
        
        Args:
            last_turn: The previous ConversationTurn
            
        Returns:
            True if reformatting is possible, False otherwise
        """
        if not last_turn:
            return False

        if not hasattr(last_turn, 'answer') or not last_turn.answer:
            return False

        # Allow nested reformatting (e.g., "explain more" -> "list in bullet points"),
        # but block placeholder responses.
        if hasattr(last_turn, 'query_type') and last_turn.query_type == "formatting_request":
            if last_turn.answer.strip().lower().startswith("there is no previous answer to reformat"):
                logger.warning("Last turn was placeholder reformatting response; skipping nested reformatting")
                return False

        return True
