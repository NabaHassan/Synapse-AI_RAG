"""
Email Metadata Extractor for Enhanced Document Indexing.

Extracts email metadata (sender, receiver, date, subject) from document text
using regex patterns. Works on all document types including PDFs containing
printed email content.

Patterns recognized:
  - From: / Sender: / Sent By:
  - To: / Recipient: / Sent To:
  - Date: / Sent: / Sent On:
  - Subject: / Re: / Fwd:
"""

import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailMetadataExtractor:
    """
    Extracts email metadata from document text using regex patterns.

    Designed to work on:
    - Plain text email files (.eml, .txt)
    - PDFs containing printed/scanned email content
    - Any document with email-like header formatting

    Usage:
        extractor = EmailMetadataExtractor()
        metadata = extractor.extract("From: john@example.com\\nTo: jane@example.com\\n...")
    """

    # Regex patterns for email headers
    # Each pattern group tries multiple common formats

    SENDER_PATTERNS = [
        # "From: John Doe <john@example.com>" or "From: john@example.com"
        r'(?:^|\n)\s*(?:From|Sender|Sent\s*By)\s*:\s*(.+?)(?:\n|$)',
    ]

    RECEIVER_PATTERNS = [
        # "To: Jane Doe <jane@example.com>" or "To: jane@example.com"
        r'(?:^|\n)\s*(?:To|Recipient|Sent\s*To)\s*:\s*(.+?)(?:\n|$)',
    ]

    DATE_PATTERNS = [
        # "Date: March 15, 2015" or "Sent: 2015-03-15" or "Date: 03/15/2015"
        r'(?:^|\n)\s*(?:Date|Sent|Sent\s*On|Received)\s*:\s*(.+?)(?:\n|$)',
    ]

    SUBJECT_PATTERNS = [
        # "Subject: Meeting arrangement" or "Re: Meeting arrangement"
        r'(?:^|\n)\s*(?:Subject|Subj|Re|Fwd)\s*:\s*(.+?)(?:\n|$)',
    ]

    # Email address regex for validation
    EMAIL_ADDRESS_RE = re.compile(
        r'[\w.+-]+@[\w-]+\.[\w.-]+'
    )

    def __init__(self):
        """Initialize the email metadata extractor."""
        # Pre-compile all patterns
        self._sender_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.SENDER_PATTERNS]
        self._receiver_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.RECEIVER_PATTERNS]
        self._date_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.DATE_PATTERNS]
        self._subject_patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.SUBJECT_PATTERNS]

        logger.info("EmailMetadataExtractor initialized")

    def extract(self, text: str) -> Dict[str, any]:
        """
        Extract email metadata from text.

        Args:
            text: Document text to extract email metadata from

        Returns:
            {
                "email_sender": "john@example.com" or "John Doe <john@example.com>",
                "email_receiver": "jane@example.com" or "Jane Doe <jane@example.com>",
                "email_date": "2015-03-15" or raw date string,
                "email_subject": "Meeting arrangement",
                "is_email": True/False
            }
            Empty strings for fields that couldn't be extracted.
        """
        if not text or not text.strip():
            return self._empty_result()

        # Only check the first ~3000 chars for headers (they should be near the top)
        header_text = text[:3000]

        sender = self._extract_field(header_text, self._sender_patterns)
        receiver = self._extract_field(header_text, self._receiver_patterns)
        date = self._extract_field(header_text, self._date_patterns)
        subject = self._extract_field(header_text, self._subject_patterns)

        # Determine if this is an email document
        # Require at least sender OR (receiver AND subject) to classify as email
        is_email = bool(sender) or (bool(receiver) and bool(subject))

        # Clean up extracted values
        sender = self._clean_header_value(sender)
        receiver = self._clean_header_value(receiver)
        date = self._normalize_date(date) if date else ""
        subject = self._clean_header_value(subject)

        result = {
            "email_sender": sender,
            "email_receiver": receiver,
            "email_date": date,
            "email_subject": subject,
            "is_email": is_email,
        }

        if is_email:
            logger.debug(
                f"Email detected - From: {sender[:50]}, "
                f"To: {receiver[:50]}, "
                f"Subject: {subject[:50]}"
            )

        return result

    def extract_multiple(self, text: str) -> List[Dict[str, any]]:
        """
        Extract metadata from documents containing multiple emails.

        Splits text on common email delimiters and extracts metadata
        from each segment.

        Args:
            text: Document text potentially containing multiple emails

        Returns:
            List of email metadata dicts (same structure as extract())
        """
        if not text or not text.strip():
            return []

        # Common email thread delimiters
        delimiters = [
            r'\n-{3,}\s*(?:Original\s+Message|Forwarded\s+Message)\s*-{3,}\n',
            r'\n_{3,}\n',
            r'\nOn\s+.+?\s+wrote:\s*\n',
            r'\n\s*(?:From|Sender)\s*:\s*.+?\n\s*(?:Sent|Date)\s*:\s*.+?\n',
        ]

        # Try to split on delimiters
        segments = [text]
        for delimiter in delimiters:
            new_segments = []
            for segment in segments:
                parts = re.split(delimiter, segment, flags=re.IGNORECASE)
                new_segments.extend(parts)
            if len(new_segments) > len(segments):
                segments = new_segments
                break  # Use first delimiter that works

        # Extract metadata from each segment
        results = []
        for segment in segments:
            if len(segment.strip()) < 20:
                continue
            metadata = self.extract(segment)
            if metadata["is_email"]:
                results.append(metadata)

        return results if results else [self.extract(text)]

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _extract_field(text: str, patterns: List[re.Pattern]) -> str:
        """Try each pattern and return the first match."""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return ""

    @staticmethod
    def _clean_header_value(value: str) -> str:
        """Clean up an extracted header value."""
        if not value:
            return ""

        # Remove trailing punctuation artifacts
        value = value.rstrip(';,')

        # Collapse whitespace
        value = re.sub(r'\s+', ' ', value).strip()

        # Truncate if unreasonably long (likely a parsing error)
        if len(value) > 200:
            value = value[:200]

        return value

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """
        Try to normalize a date string to ISO format (YYYY-MM-DD).
        Falls back to returning the cleaned original string.
        """
        if not date_str:
            return ""

        date_str = date_str.strip()

        # Common date formats to try
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%a, %d %b %Y %H:%M:%S",
            "%A, %B %d, %Y",
            "%m-%d-%Y",
        ]

        # Strip timezone info and extra text for parsing
        clean_date = re.sub(r'\s*[\+\-]\d{4}.*$', '', date_str)
        clean_date = re.sub(r'\s*\(.*?\)\s*$', '', clean_date)
        clean_date = clean_date.strip()

        for fmt in formats:
            try:
                parsed = datetime.strptime(clean_date, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Could not parse; return cleaned original
        return re.sub(r'\s+', ' ', date_str).strip()[:50]

    @staticmethod
    def _empty_result() -> Dict[str, any]:
        """Return an empty email metadata result."""
        return {
            "email_sender": "",
            "email_receiver": "",
            "email_date": "",
            "email_subject": "",
            "is_email": False,
        }
