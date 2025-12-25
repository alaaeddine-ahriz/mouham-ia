"""PDF text extraction with layout preservation and bounding boxes.

Extracts text from PDFs with line-level granularity, preserving:
- Page numbers
- Bounding boxes for each line (for UI highlighting)
- Reading order (handles multi-column layouts)
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

import pdfplumber
from pdfplumber.page import Page

from ..config import get_category_from_path


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class BBox:
    """Bounding box for a text element (coordinates in PDF points)."""
    
    x0: float  # Left
    y0: float  # Top (from page top)
    x1: float  # Right
    y1: float  # Bottom
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}
    
    @classmethod
    def from_word(cls, word: dict) -> "BBox":
        """Create from pdfplumber word dict."""
        return cls(
            x0=word["x0"],
            y0=word["top"],
            x1=word["x1"],
            y1=word["bottom"],
        )
    
    def merge(self, other: "BBox") -> "BBox":
        """Merge two bboxes into one that contains both."""
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


@dataclass
class LineBlock:
    """A single line of text with its position."""
    
    text: str
    page: int  # 1-indexed page number
    bbox: BBox
    
    def to_highlight(self) -> dict:
        """Convert to highlight format for UI."""
        return {
            "page": self.page,
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class ExtractedPage:
    """Content extracted from a single PDF page with layout."""
    
    page_number: int  # 1-indexed
    lines: list[LineBlock]
    width: float
    height: float


@dataclass
class ExtractedDocument:
    """Full document with line-level extraction and layout info."""
    
    source: str  # Filename
    pages: list[ExtractedPage]
    total_pages: int
    category: str = "general"
    file_path: str = ""
    
    @property
    def all_lines(self) -> list[LineBlock]:
        """Get all lines across all pages in reading order."""
        lines = []
        for page in self.pages:
            lines.extend(page.lines)
        return lines
    
    @property
    def full_text(self) -> str:
        """Get full document text from all lines."""
        return "\n".join(line.text for line in self.all_lines)


# ==============================================================================
# Legacy Data Classes (for backward compatibility)
# ==============================================================================


@dataclass
class PageContent:
    """Content extracted from a single PDF page (legacy format)."""
    
    page_number: int  # 1-indexed
    text: str
    source: str  # filename
    category: str = "general"


@dataclass
class DocumentContent:
    """Full document content (legacy format for contracts)."""
    
    source: str
    pages: list[PageContent]
    total_pages: int
    category: str = "general"
    file_path: str = ""
    
    @property
    def full_text(self) -> str:
        """Get the full document text."""
        return "\n\n".join(page.text for page in self.pages)


# ==============================================================================
# Extraction Functions
# ==============================================================================


def _group_words_into_lines(
    words: list[dict],
    line_tolerance: float = 3.0,
) -> list[list[dict]]:
    """
    Group words into lines based on vertical position.
    
    Args:
        words: List of word dicts from pdfplumber
        line_tolerance: Max vertical distance to consider same line
        
    Returns:
        List of word groups, each representing a line
    """
    if not words:
        return []
    
    # Sort by vertical position (top), then horizontal (x0)
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    
    lines: list[list[dict]] = []
    current_line: list[dict] = [sorted_words[0]]
    current_top = sorted_words[0]["top"]
    
    for word in sorted_words[1:]:
        # Check if word is on the same line (within tolerance)
        if abs(word["top"] - current_top) <= line_tolerance:
            current_line.append(word)
        else:
            # Start new line
            # Sort current line by x position before saving
            current_line.sort(key=lambda w: w["x0"])
            lines.append(current_line)
            current_line = [word]
            current_top = word["top"]
    
    # Don't forget the last line
    if current_line:
        current_line.sort(key=lambda w: w["x0"])
        lines.append(current_line)
    
    return lines


def _detect_columns(
    lines: list[list[dict]],
    page_width: float,
    gap_threshold: float = 50.0,
) -> list[list[list[dict]]]:
    """
    Detect multi-column layout and split lines by column.
    
    Args:
        lines: List of word groups (lines)
        page_width: Width of the page
        gap_threshold: Minimum horizontal gap to detect column break
        
    Returns:
        List of columns, each containing lines belonging to that column
    """
    if not lines:
        return []
    
    # Analyze x-positions to detect column boundaries
    # Look for consistent gaps in the middle of the page
    all_x_positions = []
    for line in lines:
        for word in line:
            all_x_positions.append(word["x0"])
            all_x_positions.append(word["x1"])
    
    if not all_x_positions:
        return [lines]
    
    min_x = min(all_x_positions)
    max_x = max(all_x_positions)
    
    # Simple heuristic: check if there's a consistent gap in the middle third
    middle_start = min_x + (max_x - min_x) * 0.3
    middle_end = min_x + (max_x - min_x) * 0.7
    
    # Check each line for gap in middle region
    gap_detected = False
    column_boundary = page_width / 2
    
    for line in lines:
        if len(line) < 2:
            continue
        
        # Check for large gaps between consecutive words
        for i in range(len(line) - 1):
            gap_start = line[i]["x1"]
            gap_end = line[i + 1]["x0"]
            gap = gap_end - gap_start
            
            if gap > gap_threshold and middle_start < gap_start < middle_end:
                gap_detected = True
                column_boundary = (gap_start + gap_end) / 2
                break
        
        if gap_detected:
            break
    
    if not gap_detected:
        # Single column
        return [lines]
    
    # Split lines into two columns
    left_column: list[list[dict]] = []
    right_column: list[list[dict]] = []
    
    for line in lines:
        left_words = [w for w in line if w["x1"] < column_boundary]
        right_words = [w for w in line if w["x0"] > column_boundary]
        
        if left_words:
            left_column.append(left_words)
        if right_words:
            right_column.append(right_words)
    
    return [left_column, right_column]


def _words_to_line_block(words: list[dict], page_num: int) -> LineBlock:
    """Convert a list of words to a LineBlock."""
    # Merge all word bboxes
    bbox = BBox.from_word(words[0])
    for word in words[1:]:
        bbox = bbox.merge(BBox.from_word(word))
    
    # Join word texts with spaces
    text = " ".join(w["text"] for w in words)
    
    return LineBlock(text=text, page=page_num, bbox=bbox)


def _extract_page_lines(page: Page, page_num: int) -> list[LineBlock]:
    """
    Extract lines from a single page with layout preservation.
    
    Handles multi-column layouts by detecting columns and
    returning lines in proper reading order.
    """
    # Extract words with their positions
    words = page.extract_words(
        keep_blank_chars=True,
        x_tolerance=3,
        y_tolerance=3,
    )
    
    if not words:
        return []
    
    # Group into lines
    lines = _group_words_into_lines(words)
    
    # Detect and handle columns
    columns = _detect_columns(lines, page.width)
    
    # Build LineBlocks in reading order (left column first, then right)
    line_blocks: list[LineBlock] = []
    for column in columns:
        for word_group in column:
            if word_group:
                line_blocks.append(_words_to_line_block(word_group, page_num))
    
    return line_blocks


def _is_header_or_footer(
    line: LineBlock,
    page_height: float,
    margin: float = 50.0,
) -> bool:
    """Check if a line is likely a header or footer based on position."""
    # Check if line is in the top or bottom margin
    if line.bbox.y0 < margin:  # Near top
        return True
    if line.bbox.y1 > page_height - margin:  # Near bottom
        return True
    return False


def _clean_lines(lines: list[LineBlock], page_height: float) -> list[LineBlock]:
    """
    Clean extracted lines:
    - Remove headers/footers
    - Dehyphenate line-end hyphens
    """
    cleaned: list[LineBlock] = []
    
    for i, line in enumerate(lines):
        # Skip likely headers/footers
        if _is_header_or_footer(line, page_height):
            # But keep if it looks like article header
            if not re.match(r"^(Article|Art\.)\s+\d+", line.text, re.IGNORECASE):
                continue
        
        # Dehyphenate: if line ends with hyphen and next line exists
        if line.text.endswith("-") and i < len(lines) - 1:
            # Merge with start of next line
            next_line = lines[i + 1]
            merged_text = line.text[:-1] + next_line.text.split()[0] if next_line.text else line.text[:-1]
            
            # Keep original bbox for now (highlighting will use the original)
            cleaned.append(LineBlock(
                text=merged_text,
                page=line.page,
                bbox=line.bbox,
            ))
        else:
            cleaned.append(line)
    
    return cleaned


# ==============================================================================
# Public API
# ==============================================================================


def extract_pdf_with_layout(
    file_path: Path,
    category: str | None = None,
) -> ExtractedDocument:
    """
    Extract text from a PDF with layout preservation.
    
    Returns lines with bounding boxes for highlighting support.
    
    Args:
        file_path: Path to the PDF file
        category: Optional category override
        
    Returns:
        ExtractedDocument with line-level extraction
    """
    pages: list[ExtractedPage] = []
    source = file_path.name
    
    if category is None:
        category = get_category_from_path(str(file_path))
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            lines = _extract_page_lines(page, page_num)
            
            # Clean lines (remove headers/footers, dehyphenate)
            lines = _clean_lines(lines, page.height)
            
            if lines:
                pages.append(ExtractedPage(
                    page_number=page_num,
                    lines=lines,
                    width=page.width,
                    height=page.height,
                ))
        
        return ExtractedDocument(
            source=source,
            pages=pages,
            total_pages=len(pdf.pages),
            category=category,
            file_path=str(file_path),
        )


def extract_pdfs_with_layout(
    directory: Path,
    category: str | None = None,
    recursive: bool = True,
) -> list[ExtractedDocument]:
    """
    Extract text from all PDFs in a directory with layout preservation.
    
    Args:
        directory: Path to directory containing PDFs
        category: Optional category override
        recursive: If True, search subdirectories
        
    Returns:
        List of ExtractedDocument objects
    """
    documents: list[ExtractedDocument] = []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    
    for pdf_path in sorted(directory.glob(pattern)):
        try:
            doc = extract_pdf_with_layout(pdf_path, category=category)
            documents.append(doc)
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")
    
    return documents


# ==============================================================================
# Legacy API (for backward compatibility with contracts)
# ==============================================================================


def extract_pdf(file_path: Path, category: str | None = None) -> DocumentContent:
    """
    Extract text from a PDF file (legacy format).
    
    Kept for backward compatibility with contract ingestion.
    """
    pages: list[PageContent] = []
    source = file_path.name
    
    if category is None:
        category = get_category_from_path(str(file_path))
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
            
            if text:
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    source=source,
                    category=category,
                ))
        
        return DocumentContent(
            source=source,
            pages=pages,
            total_pages=len(pdf.pages),
            category=category,
            file_path=str(file_path),
        )


def extract_pdfs_from_directory(
    directory: Path,
    category: str | None = None,
    recursive: bool = True,
) -> list[DocumentContent]:
    """
    Extract text from all PDFs in a directory (legacy format).
    
    Kept for backward compatibility with contract ingestion.
    """
    documents: list[DocumentContent] = []
    pattern = "**/*.pdf" if recursive else "*.pdf"
    
    for pdf_path in sorted(directory.glob(pattern)):
        try:
            doc = extract_pdf(pdf_path, category=category)
            documents.append(doc)
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")
    
    return documents
