"""PDF text extraction with page-level tracking."""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""

    page_number: int  # 1-indexed
    text: str
    source: str  # filename


@dataclass
class DocumentContent:
    """Full document content with page-level granularity."""

    source: str
    pages: list[PageContent]
    total_pages: int

    @property
    def full_text(self) -> str:
        """Get the full document text."""
        return "\n\n".join(page.text for page in self.pages)


def extract_pdf(file_path: Path) -> DocumentContent:
    """
    Extract text from a PDF file with page-level tracking.

    Args:
        file_path: Path to the PDF file

    Returns:
        DocumentContent with page-by-page text extraction
    """
    pages: list[PageContent] = []
    source = file_path.name

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
                    )
                )

        return DocumentContent(
            source=source,
            pages=pages,
            total_pages=len(pdf.pages),
        )


def extract_pdfs_from_directory(directory: Path) -> list[DocumentContent]:
    """
    Extract text from all PDF files in a directory.

    Args:
        directory: Path to directory containing PDFs

    Returns:
        List of DocumentContent objects
    """
    documents: list[DocumentContent] = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        try:
            doc = extract_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {e}")

    return documents

