"""Smart text chunking with metadata preservation."""

import re
from dataclasses import dataclass

import tiktoken

from .pdf_extractor import DocumentContent, PageContent


@dataclass
class Chunk:
    """A text chunk with full metadata for citation."""

    text: str  # The exact verbatim text
    source: str  # Original filename
    page_number: int  # Starting page number
    chunk_index: int  # Position in document
    section_header: str | None  # Extracted section title if found
    category: str = "general"  # Legal domain category

    def to_metadata(self) -> dict:
        """Convert to Pinecone metadata dict."""
        return {
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "section_header": self.section_header or "",
            "category": self.category,
        }


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def _extract_section_header(text: str) -> str | None:
    """
    Try to extract a section header from the start of text.

    Looks for patterns like:
    - "Article 1234" or "Article 1234-5"
    - "Section 1.2"
    - "CHAPTER IV"
    - Lines that are ALL CAPS and short
    """
    lines = text.strip().split("\n")
    if not lines:
        return None

    first_line = lines[0].strip()

    # Check for article/section patterns
    patterns = [
        r"^(Article\s+[\d\-]+)",
        r"^(Section\s+[\d\.]+)",
        r"^(CHAPTER\s+[IVXLCDM\d]+)",
        r"^(Chapitre\s+[IVXLCDM\d]+)",
        r"^(Titre\s+[IVXLCDM\d]+)",
    ]

    for pattern in patterns:
        match = re.match(pattern, first_line, re.IGNORECASE)
        if match:
            return match.group(1)

    # Check for short ALL CAPS lines (likely headers)
    if first_line.isupper() and len(first_line) < 100:
        return first_line

    return None


def chunk_document(
    document: DocumentContent,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Chunk a document with smart boundaries and metadata.

    Args:
        document: The extracted document content
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of Chunk objects with metadata
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    chunks: list[Chunk] = []
    chunk_index = 0

    # Process page by page to maintain page tracking
    for page in document.pages:
        paragraphs = _split_into_paragraphs(page.text)
        current_chunk_text = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para, encoding)

            # If single paragraph exceeds chunk size, split it
            if para_tokens > chunk_size:
                # Flush current chunk if not empty
                if current_chunk_text.strip():
                    chunks.append(
                        Chunk(
                            text=current_chunk_text.strip(),
                            source=document.source,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            section_header=_extract_section_header(current_chunk_text),
                            category=document.category,
                        )
                    )
                    chunk_index += 1
                    current_chunk_text = ""
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = _split_into_sentences(para)
                for sentence in sentences:
                    sent_tokens = _count_tokens(sentence, encoding)
                    if current_tokens + sent_tokens > chunk_size and current_chunk_text:
                        chunks.append(
                            Chunk(
                                text=current_chunk_text.strip(),
                                source=document.source,
                                page_number=page.page_number,
                                chunk_index=chunk_index,
                                section_header=_extract_section_header(
                                    current_chunk_text
                                ),
                                category=document.category,
                            )
                        )
                        chunk_index += 1
                        # Keep overlap
                        overlap_text = _get_overlap_text(
                            current_chunk_text, chunk_overlap, encoding
                        )
                        current_chunk_text = overlap_text + " " + sentence
                        current_tokens = _count_tokens(current_chunk_text, encoding)
                    else:
                        current_chunk_text += " " + sentence
                        current_tokens += sent_tokens
            # Normal case: add paragraph to current chunk
            elif current_tokens + para_tokens > chunk_size and current_chunk_text:
                chunks.append(
                    Chunk(
                        text=current_chunk_text.strip(),
                        source=document.source,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        section_header=_extract_section_header(current_chunk_text),
                        category=document.category,
                    )
                )
                chunk_index += 1
                # Keep overlap
                overlap_text = _get_overlap_text(
                    current_chunk_text, chunk_overlap, encoding
                )
                current_chunk_text = overlap_text + "\n\n" + para
                current_tokens = _count_tokens(current_chunk_text, encoding)
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                current_tokens += para_tokens

        # Flush remaining content from page
        if current_chunk_text.strip():
            chunks.append(
                Chunk(
                    text=current_chunk_text.strip(),
                    source=document.source,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    section_header=_extract_section_header(current_chunk_text),
                    category=document.category,
                )
            )
            chunk_index += 1

    return chunks


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    # Split on double newlines or single newlines followed by indent/caps
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting - handles common cases
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _get_overlap_text(text: str, overlap_tokens: int, encoding: tiktoken.Encoding) -> str:
    """Get the last N tokens of text for overlap."""
    tokens = encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    overlap_token_ids = tokens[-overlap_tokens:]
    return encoding.decode(overlap_token_ids)

