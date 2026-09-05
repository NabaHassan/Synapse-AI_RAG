"""Reusable text cleaning for online ingestion."""

from __future__ import annotations

import logging
import re
from typing import Dict, Tuple

try:
    import ftfy  # type: ignore
except ImportError:  # pragma: no cover - depends on deployment extras
    ftfy = None

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Production-ready text cleaner with comprehensive cleaning rules.
    (Identical to clean_text.py)
    """

    # Common academic/research paper artifacts to remove
    NOISE_PATTERNS = [
        # Page numbers (various formats)
        r'\n\s*\d+\s*\n',
        r'\n\s*-\s*\d+\s*-\s*\n',
        r'\n\s*Page\s+\d+\s*\n',

        # Headers and footers (common patterns)
        r'(?i)\n\s*©\s*\d{4}.*?\n',
        r'(?i)\n\s*Copyright\s+©.*?\n',
        r'(?i)\n\s*All rights reserved.*?\n',

        # URLs in headers/footers
        r'\n\s*https?://\S+\s*\n',
        r'\n\s*www\.\S+\s*\n',

        # Document metadata lines
        r'(?i)\n\s*published:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\n',
        r'(?i)\n\s*received:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*\n',

        # Excessive whitespace
        r'\n\s*\n\s*\n+',  # Multiple blank lines
        r'[ \t]+\n',  # Trailing spaces
        r'\n[ \t]+',  # Leading spaces on new lines (excessive)

        # Watermarks
        r'(?i)\b(draft|confidential|internal use only)\b\s*\n',
    ]

    # Patterns for PII detection (basic)
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    }

    def __init__(
            self,
            remove_urls: bool = False,
            remove_emails: bool = False,
            fix_unicode: bool = True,
            normalize_whitespace: bool = True,
            remove_special_chars: bool = False,
            min_word_length: int = 2,
            max_word_length: int = 45
    ):
        """Initialize text cleaner with configuration."""
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.fix_unicode = fix_unicode
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

        # Compile regex patterns for efficiency
        self.noise_patterns_compiled = [
            re.compile(pattern) for pattern in self.NOISE_PATTERNS
        ]
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(self.PII_PATTERNS['email'])

    def clean_text(self, text: str, source_file: str = "") -> Tuple[str, Dict]:
        """Clean text with all configured rules."""
        if not text or not text.strip():
            return "", {"empty_input": True}

        original_length = len(text)
        stats = {
            "original_length": original_length,
            "original_words": len(text.split()),
        }

        # Step 1: Fix encoding issues
        if self.fix_unicode:
            text = self._fix_encoding(text)
            stats["encoding_fixed"] = True

        # Step 2: Remove noise patterns
        text, noise_removals = self._remove_noise(text)
        stats["noise_patterns_removed"] = noise_removals

        # Step 3: Remove URLs
        if self.remove_urls:
            text, url_count = self._remove_urls(text)
            stats["urls_removed"] = url_count

        # Step 4: Remove emails
        if self.remove_emails:
            text, email_count = self._remove_emails(text)
            stats["emails_removed"] = email_count

        # Step 5: Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
            stats["whitespace_normalized"] = True

        # Step 6: Remove special characters (optional)
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
            stats["special_chars_removed"] = True

        # Step 7: Filter words by length
        text = self._filter_words(text)
        stats["words_filtered"] = True

        # Step 8: Remove duplicate consecutive lines
        text = self._remove_duplicate_lines(text)
        stats["duplicate_lines_removed"] = True

        # Final statistics
        stats["final_length"] = len(text)
        stats["final_words"] = len(text.split())
        stats["reduction_ratio"] = 1 - (len(text) / original_length) if original_length > 0 else 0

        return text, stats

    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Fix Unicode encoding issues using ftfy."""
        if ftfy is None:
            return text

        try:
            text = ftfy.fix_text(text)
        except Exception:
            pass
        return text

    def _remove_noise(self, text: str) -> Tuple[str, int]:
        """Remove common noise patterns."""
        removals = 0
        for pattern in self.noise_patterns_compiled:
            matches = len(pattern.findall(text))
            if matches > 0:
                text = pattern.sub('\n', text)
                removals += matches
        return text, removals

    def _remove_urls(self, text: str) -> Tuple[str, int]:
        """Remove URLs from text."""
        urls = self.url_pattern.findall(text)
        text = self.url_pattern.sub(' ', text)
        return text, len(urls)

    def _remove_emails(self, text: str) -> Tuple[str, int]:
        """Remove email addresses from text."""
        emails = self.email_pattern.findall(text)
        text = self.email_pattern.sub('[EMAIL]', text)
        return text, len(emails)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace and line breaks."""
        text = text.replace('\t', ' ')
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = text.strip()
        return text

    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """Remove special characters, keeping alphanumeric and basic punctuation."""
        text = re.sub(r'[^\w\s.,;:!?\-\'\"()\[\]{}]', '', text)
        return text

    def _filter_words(self, text: str) -> str:
        """Filter words by length to remove gibberish."""
        words = text.split()
        filtered_words = [
            word for word in words
            if self.min_word_length <= len(word) <= self.max_word_length
        ]
        return ' '.join(filtered_words)

    @staticmethod
    def _remove_duplicate_lines(text: str) -> str:
        """Remove consecutive duplicate lines."""
        lines = text.split('\n')
        deduplicated = []
        prev_line = None

        for line in lines:
            line_stripped = line.strip()
            if line_stripped != prev_line:
                deduplicated.append(line)
                prev_line = line_stripped

        return '\n'.join(deduplicated)

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect text language (simple heuristic)."""
        common_english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do',
            'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say'
        }

        words = text.lower().split()[:200]
        word_set = set(words)

        english_matches = len(word_set & common_english_words)

        if english_matches >= 5:
            return 'en'
        else:
            return 'unknown'

    @staticmethod
    def quality_score(text: str) -> float:
        """Calculate text quality score (0-1)."""
        if not text or len(text) < 50:
            return 0.0

        score = 0.0

        # Length score (0-0.3)
        length_score = min(len(text) / 5000, 1.0) * 0.3
        score += length_score

        # Word count score (0-0.2)
        words = text.split()
        word_count_score = min(len(words) / 1000, 1.0) * 0.2
        score += word_count_score

        # Average word length (0-0.2)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        if 4 <= avg_word_length <= 7:
            score += 0.2
        elif 3 <= avg_word_length <= 8:
            score += 0.1

        # Sentence count (0-0.2)
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        sentence_score = min(sentence_endings / 50, 1.0) * 0.2
        score += sentence_score

        # Punctuation presence (0-0.1)
        punct_count = sum(1 for char in text if char in '.,;:!?')
        if punct_count > len(text) * 0.01:
            score += 0.1

        return min(score, 1.0)
