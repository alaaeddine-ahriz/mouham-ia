"""PDF text extraction with page-level tracking."""

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from ..config import get_category_from_path


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    source: str  # filename
    category: str = "general"  # Legal domain category


@dataclass
class DocumentContent:
    """Full document content with page-level granularity."""

    source: str
    pages: list[PageContent]
    total_pages: int
    category: str = "general"  # Legal domain category
    file_path: str = ""  # Full path for reference

    @property
    def full_text(self) -> str:
        """Get the full document text."""
        return "\n\n".join(page.text for page in self.pages)


def extract_pdf(file_path: Path, category: str | None = None) -> DocumentContent:
    """
    Extract text from a PDF file with page-level tracking.

    Args:
        file_path: Path to the PDF file
        category: Optional category override. If None, extracted from path.

    Returns:
        DocumentContent with page-by-page text extraction
    """
    pages: list[PageContent] = []
    source = file_path.name
    
    # Determine category from path if not provided
    if category is None:
        category = get_category_from_path(str(file_path))

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Clean up whitespace while preserving paragraph structure
            text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

            if text:  # Only add pages with content
                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=text,
                        source=source,
                        category=category,
                    )
                )

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
    Extract text from all PDF files in a directory.

    Args:
        directory: Path to directory containing PDFs
        category: Optional category override. If None, extracted from each file's path.
        recursive: If True, search subdirectories (for category folders)

    Returns:
        List of DocumentContent objects
    """
    documents: list[DocumentContent] = []

    # Use recursive glob if enabled (for category subfolders)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    
    for pdf_path in sorted(directory.glob(pattern)):
        try:
            doc = extract_pdf(pdf_path, category=category)
            documents.append(doc)
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")

    return documents

