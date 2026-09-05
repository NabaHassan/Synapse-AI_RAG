"""Reusable document extraction for online ingestion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

try:
    import fitz  # type: ignore  # PyMuPDF
except ImportError:  # pragma: no cover - depends on deployment extras
    fitz = None

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """
    Universal document extractor supporting multiple file formats.
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        'pdf': 'extract_pdf',
        'txt': 'extract_txt',
        'md': 'extract_markdown',
        'docx': 'extract_docx',
        'doc': 'extract_doc',
        'csv': 'extract_csv',
        'xlsx': 'extract_excel',
        'xls': 'extract_excel',
        'json': 'extract_json',
        'html': 'extract_html',
        'htm': 'extract_html',
    }

    def __init__(self, use_unstructured_fallback: bool = True):
        """
        Initialize document extractor.
        
        Args:
            use_unstructured_fallback: Whether to use Unstructured.io as fallback for PDFs
        """
        self.use_unstructured_fallback = use_unstructured_fallback
        self.unstructured_available = False

        # Try to import optional dependencies
        self._import_optional_dependencies()

    def _import_optional_dependencies(self):
        """Import optional dependencies for various file formats."""
        
        # For PDF fallback
        if self.use_unstructured_fallback:
            try:
                from unstructured.partition.pdf import partition_pdf
                self.partition_pdf = partition_pdf
                self.unstructured_available = True
                logger.info("Unstructured.io available for PDF fallback")
            except ImportError:
                logger.warning(
                    "Unstructured.io not available. Install with: "
                    "pip install unstructured[pdf]"
                )

        # For DOCX files
        try:
            import docx
            self.docx = docx
            self.docx_available = True
        except ImportError:
            self.docx_available = False
            logger.warning("python-docx not available. Install with: pip install python-docx")

        # For Excel files
        try:
            import openpyxl
            import pandas as pd
            self.openpyxl = openpyxl
            self.pandas = pd
            self.excel_available = True
        except ImportError:
            self.excel_available = False
            logger.warning("openpyxl/pandas not available. Install with: pip install openpyxl pandas")

        # For HTML files
        try:
            from bs4 import BeautifulSoup
            self.BeautifulSoup = BeautifulSoup
            self.html_available = True
        except ImportError:
            self.html_available = False
            logger.warning("beautifulsoup4 not available. Install with: pip install beautifulsoup4")

        # For DOC files (older format)
        try:
            import textract
            self.textract = textract
            self.textract_available = True
        except ImportError:
            self.textract_available = False
            logger.debug("textract not available for .doc files")

    def extract(self, file_path: Path) -> Dict:
        """
        Extract text from any supported document format.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dict containing extracted data with unified structure
        """
        extension = file_path.suffix.lower().lstrip('.')
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            return self._error_result(
                f"Unsupported file format: .{extension}",
                extraction_method="unsupported"
            )

        method_name = self.SUPPORTED_EXTENSIONS[extension]
        method = getattr(self, method_name, None)

        if not method:
            return self._error_result(
                f"Extraction method not found: {method_name}",
                extraction_method=method_name
            )

        try:
            logger.info(f"Extracting {file_path.name} using {method_name}")
            return method(file_path)
        except Exception as e:
            logger.error(f"Extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method=method_name)

    # ==================== PDF EXTRACTION ====================
    
    def extract_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract text from PDF using PyMuPDF (with fallback).
        Reuses the logic from extract_pdfs.py
        """
        if fitz is None:
            if self.unstructured_available:
                return self._extract_pdf_with_unstructured(pdf_path)
            return self._error_result(
                "PyMuPDF is not installed. Install with: pip install PyMuPDF",
                extraction_method="pymupdf"
            )

        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            all_text = []

            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc)
            }

            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                blocks = page.get_text("blocks")
                sections = self._detect_sections(blocks)

                page_data = {
                    "page_num": page_num + 1,
                    "content": text,
                    "sections": sections,
                    "char_count": len(text),
                    "has_images": len(page.get_images()) > 0,
                    "has_tables": self._detect_tables(page)
                }

                pages_data.append(page_data)
                all_text.append(text)

            doc.close()

            return {
                "full_text": "\n".join(all_text),
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": metadata,
                "extraction_method": "pymupdf",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path.name}: {e}")
            
            # Try fallback if available
            if self.unstructured_available:
                return self._extract_pdf_with_unstructured(pdf_path)
            
            return self._error_result(str(e), extraction_method="pymupdf")

    def _extract_pdf_with_unstructured(self, pdf_path: Path) -> Dict:
        """Fallback PDF extraction using Unstructured.io"""
        try:
            elements = self.partition_pdf(str(pdf_path))
            pages_data = {}
            all_text = []

            for element in elements:
                page_num = getattr(element.metadata, 'page_number', 1)
                
                if page_num not in pages_data:
                    pages_data[page_num] = {
                        "page_num": page_num,
                        "content": [],
                        "sections": [],
                        "char_count": 0
                    }

                text = str(element)
                pages_data[page_num]["content"].append(text)
                pages_data[page_num]["char_count"] += len(text)
                all_text.append(text)

            pages_list = []
            for page_num in sorted(pages_data.keys()):
                page = pages_data[page_num]
                page["content"] = "\n".join(page["content"])
                pages_list.append(page)

            return {
                "full_text": "\n".join(all_text),
                "page_count": len(pages_list),
                "pages": pages_list,
                "metadata": {"page_count": len(pages_list)},
                "extraction_method": "unstructured",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Unstructured extraction failed for {pdf_path.name}: {e}")
            return self._error_result(str(e), extraction_method="unstructured")

    @staticmethod
    def _detect_sections(blocks: List) -> List[Dict]:
        """Detect section headings from text blocks (for PDFs)."""
        sections = []
        for block in blocks:
            if len(block) < 7:
                continue
            text = block[4].strip()
            if text and len(text) < 100:
                if text.isupper() or text.istitle() or text.endswith(':'):
                    sections.append({
                        "title": text,
                        "position": block[1]
                    })
        return sections

    @staticmethod
    def _detect_tables(page) -> bool:
        """Detect if page contains tables (for PDFs)."""
        try:
            drawings = page.get_drawings()
            if len(drawings) > 20:
                return True
            return False
        except:
            return False

    # ==================== TEXT FILE EXTRACTION ====================

    def extract_txt(self, file_path: Path) -> Dict:
        """Extract text from plain text files (.txt)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Split into pages (simulate with line-based chunks)
            lines = text.split('\n')
            pages_data = self._create_page_chunks(lines, lines_per_page=50)

            return {
                "full_text": text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": {
                    "file_type": "text",
                    "encoding": "utf-8",
                    "line_count": len(lines)
                },
                "extraction_method": "text",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Text extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="text")

    # ==================== MARKDOWN EXTRACTION ====================

    def extract_markdown(self, file_path: Path) -> Dict:
        """Extract text from Markdown files (.md)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            # Detect sections from markdown headers
            lines = text.split('\n')
            sections = []
            for line in lines:
                if line.startswith('#'):
                    sections.append({
                        "title": line.lstrip('#').strip(),
                        "level": len(line) - len(line.lstrip('#'))
                    })

            pages_data = self._create_page_chunks(lines, lines_per_page=50)
            
            # Add sections to first page
            if pages_data:
                pages_data[0]["sections"] = sections

            return {
                "full_text": text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": {
                    "file_type": "markdown",
                    "section_count": len(sections)
                },
                "extraction_method": "markdown",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Markdown extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="markdown")

    # ==================== DOCX EXTRACTION ====================

    def extract_docx(self, file_path: Path) -> Dict:
        """Extract text from DOCX files."""
        if not self.docx_available:
            return self._error_result(
                "python-docx not installed. Install with: pip install python-docx",
                extraction_method="docx"
            )

        try:
            doc = self.docx.Document(file_path)
            
            # Extract paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            full_text = "\n".join(paragraphs)

            # Extract metadata
            core_props = doc.core_properties
            metadata = {
                "title": core_props.title or "",
                "author": core_props.author or "",
                "subject": core_props.subject or "",
                "created": str(core_props.created) if core_props.created else "",
                "modified": str(core_props.modified) if core_props.modified else "",
                "paragraph_count": len(paragraphs)
            }

            # Simulate pages (50 paragraphs per page)
            pages_data = []
            for i in range(0, len(paragraphs), 50):
                page_paras = paragraphs[i:i+50]
                page_text = "\n".join(page_paras)
                pages_data.append({
                    "page_num": len(pages_data) + 1,
                    "content": page_text,
                    "char_count": len(page_text),
                    "paragraph_count": len(page_paras)
                })

            if not pages_data:  # At least one page
                pages_data = [{
                    "page_num": 1,
                    "content": full_text,
                    "char_count": len(full_text),
                    "paragraph_count": len(paragraphs)
                }]

            return {
                "full_text": full_text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": metadata,
                "extraction_method": "docx",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="docx")

    # ==================== DOC EXTRACTION (legacy) ====================

    def extract_doc(self, file_path: Path) -> Dict:
        """Extract text from legacy DOC files."""
        if self.textract_available:
            try:
                text = self.textract.process(str(file_path)).decode('utf-8')
                lines = text.split('\n')
                pages_data = self._create_page_chunks(lines, lines_per_page=50)

                return {
                    "full_text": text,
                    "page_count": len(pages_data),
                    "pages": pages_data,
                    "metadata": {"file_type": "doc"},
                    "extraction_method": "textract",
                    "success": True,
                    "error": None
                }
            except Exception as e:
                logger.error(f"DOC extraction failed for {file_path.name}: {e}")
                return self._error_result(str(e), extraction_method="textract")
        else:
            return self._error_result(
                "textract not installed. Install with: pip install textract",
                extraction_method="textract"
            )

    # ==================== CSV EXTRACTION ====================

    def extract_csv(self, file_path: Path) -> Dict:
        """Extract text from CSV files."""
        if not self.excel_available:
            return self._error_result(
                "pandas not installed. Install with: pip install pandas",
                extraction_method="csv"
            )

        try:
            df = self.pandas.read_csv(file_path)
            
            # Convert to text format
            text_parts = []
            
            # Add header
            text_parts.append("Columns: " + ", ".join(df.columns))
            text_parts.append("")
            
            # Add rows
            for idx, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                text_parts.append(f"Row {idx + 1}: {row_text}")

            full_text = "\n".join(text_parts)

            # Create pages (50 rows per page)
            pages_data = []
            rows_per_page = 50
            for i in range(0, len(df), rows_per_page):
                chunk = df.iloc[i:i+rows_per_page]
                page_text_parts = ["Columns: " + ", ".join(df.columns), ""]
                for idx, row in chunk.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    page_text_parts.append(f"Row {idx + 1}: {row_text}")
                
                page_text = "\n".join(page_text_parts)
                pages_data.append({
                    "page_num": len(pages_data) + 1,
                    "content": page_text,
                    "char_count": len(page_text),
                    "row_count": len(chunk)
                })

            return {
                "full_text": full_text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": {
                    "file_type": "csv",
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns)
                },
                "extraction_method": "csv",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"CSV extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="csv")

    # ==================== EXCEL EXTRACTION ====================

    def extract_excel(self, file_path: Path) -> Dict:
        """Extract text from Excel files (.xlsx, .xls)."""
        if not self.excel_available:
            return self._error_result(
                "openpyxl/pandas not installed. Install with: pip install openpyxl pandas",
                extraction_method="excel"
            )

        try:
            # Read all sheets
            excel_file = self.pandas.ExcelFile(file_path)
            all_sheets_text = []
            
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                
                sheet_text_parts = [f"=== Sheet: {sheet_name} ===", ""]
                sheet_text_parts.append("Columns: " + ", ".join(df.columns))
                sheet_text_parts.append("")
                
                for idx, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    sheet_text_parts.append(f"Row {idx + 1}: {row_text}")
                
                all_sheets_text.append("\n".join(sheet_text_parts))

            full_text = "\n\n".join(all_sheets_text)
            
            # Create pages (each sheet or chunk as a page)
            lines = full_text.split('\n')
            pages_data = self._create_page_chunks(lines, lines_per_page=50)

            return {
                "full_text": full_text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": {
                    "file_type": "excel",
                    "sheet_count": len(excel_file.sheet_names),
                    "sheet_names": excel_file.sheet_names
                },
                "extraction_method": "excel",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"Excel extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="excel")

    # ==================== JSON EXTRACTION ====================

    def extract_json(self, file_path: Path) -> Dict:
        """Extract text from JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert JSON to readable text
            text = json.dumps(data, indent=2, ensure_ascii=False)
            
            lines = text.split('\n')
            pages_data = self._create_page_chunks(lines, lines_per_page=100)

            return {
                "full_text": text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": {
                    "file_type": "json",
                    "structure": type(data).__name__
                },
                "extraction_method": "json",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"JSON extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="json")

    # ==================== HTML EXTRACTION ====================

    def extract_html(self, file_path: Path) -> Dict:
        """Extract text from HTML files."""
        if not self.html_available:
            return self._error_result(
                "beautifulsoup4 not installed. Install with: pip install beautifulsoup4",
                extraction_method="html"
            )

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()

            soup = self.BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator='\n')
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            full_text = "\n".join(lines)

            # Extract metadata from HTML
            title = soup.find('title')
            metadata = {
                "file_type": "html",
                "title": title.string if title else "",
            }

            pages_data = self._create_page_chunks(lines, lines_per_page=50)

            return {
                "full_text": full_text,
                "page_count": len(pages_data),
                "pages": pages_data,
                "metadata": metadata,
                "extraction_method": "html",
                "success": True,
                "error": None
            }

        except Exception as e:
            logger.error(f"HTML extraction failed for {file_path.name}: {e}")
            return self._error_result(str(e), extraction_method="html")

    # ==================== HELPER METHODS ====================

    @staticmethod
    def _create_page_chunks(lines: List[str], lines_per_page: int = 50) -> List[Dict]:
        """Create page chunks from lines of text."""
        pages_data = []
        for i in range(0, len(lines), lines_per_page):
            page_lines = lines[i:i+lines_per_page]
            page_text = "\n".join(page_lines)
            pages_data.append({
                "page_num": len(pages_data) + 1,
                "content": page_text,
                "char_count": len(page_text),
                "line_count": len(page_lines)
            })
        
        if not pages_data:  # At least one page
            pages_data = [{
                "page_num": 1,
                "content": "",
                "char_count": 0,
                "line_count": 0
            }]
        
        return pages_data

    @staticmethod
    def _error_result(error_message: str, extraction_method: str = "unknown") -> Dict:
        """Create error result with standard structure."""
        return {
            "full_text": "",
            "page_count": 0,
            "pages": [],
            "metadata": {},
            "extraction_method": extraction_method,
            "success": False,
            "error": error_message
        }
