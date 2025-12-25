"""Article-centric text chunking for legal documents.

Produces one chunk per legal article, with sub-chunking for long articles.
Preserves bounding box metadata for PDF highlighting.
"""

import json
import re
from dataclasses import dataclass, field

import tiktoken

from .article_detector import Article, detect_structure
from .pdf_extractor import (
    DocumentContent,
    ExtractedDocument,
    LineBlock,
    PageContent,
)


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class ArticleChunk:
    """A chunk representing a legal article (or sub-article)."""
    
    text: str  # Clean article text
    source: str  # PDF filename
    doc_id: str  # Document identifier (derived from filename)
    article_id: str  # e.g., "Article 1134"
    article_number: str  # e.g., "1134"
    section_path: list[str]  # Hierarchical path
    page_start: int
    page_end: int
    highlights: list[dict]  # [{page, bbox}, ...] for UI highlighting
    sub_chunk_index: int = 0  # 0 if single chunk, 1+ for sub-chunks
    total_sub_chunks: int = 1  # Total number of sub-chunks for this article
    category: str = "general"
    
    def to_metadata(self) -> dict:
        """Convert to Pinecone metadata dict."""
        return {
            "text": self.text,
            "source": self.source,
            "doc_id": self.doc_id,
            "article_id": self.article_id,
            "article_number": self.article_number,
            "section_path": json.dumps(self.section_path),  # JSON string for Pinecone
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_number": self.page_start,  # For backward compatibility
            "highlights": json.dumps(self.highlights),  # JSON string for Pinecone
            "sub_chunk_index": self.sub_chunk_index,
            "total_sub_chunks": self.total_sub_chunks,
            "section_header": self.article_id,  # For backward compatibility
            "category": self.category,
        }
    
    def get_vector_id(self, namespace: str) -> str:
        """Generate a stable vector ID for Pinecone."""
        # Sanitize components for ID
        safe_doc = re.sub(r"[^a-zA-Z0-9_-]", "_", self.doc_id)
        safe_article = re.sub(r"[^a-zA-Z0-9_-]", "_", self.article_id)
        
        if self.total_sub_chunks > 1:
            return f"{namespace}_{safe_doc}_{safe_article}_{self.sub_chunk_index}"
        return f"{namespace}_{safe_doc}_{safe_article}"


@dataclass
class Chunk:
    """A text chunk with full metadata for citation (legacy format)."""

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


# ==============================================================================
# Token Counting
# ==============================================================================


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count tokens in text."""
    return len(encoding.encode(text))


def _get_encoding() -> tiktoken.Encoding:
    """Get the tokenizer encoding."""
    return tiktoken.encoding_for_model("gpt-4o")


# ==============================================================================
# Article Chunking (New Approach)
# ==============================================================================


def _split_article_into_paragraphs(article: Article) -> list[list[LineBlock]]:
    """
    Split article lines into paragraph groups.
    
    Uses blank lines or significant vertical gaps as paragraph boundaries.
    """
    if not article.lines:
        return []
    
    paragraphs: list[list[LineBlock]] = []
    current_para: list[LineBlock] = []
    
    for i, line in enumerate(article.lines):
        current_para.append(line)
        
        # Check for paragraph break conditions
        is_break = False
        
        # Blank or very short line (e.g., just punctuation)
        if len(line.text.strip()) < 3:
            is_break = True
        
        # Line ends with paragraph-ending punctuation
        if line.text.rstrip().endswith(('.', ':', ';')):
            # Check if next line starts with caps or indent
            if i + 1 < len(article.lines):
                next_line = article.lines[i + 1]
                if next_line.text and next_line.text[0].isupper():
                    is_break = True
        
        if is_break and current_para:
            paragraphs.append(current_para)
            current_para = []
    
    # Don't forget remaining lines
    if current_para:
        paragraphs.append(current_para)
    
    return paragraphs


def _create_sub_chunk(
    article: Article,
    lines: list[LineBlock],
    sub_index: int,
    total_subs: int,
    source: str,
    category: str,
) -> ArticleChunk:
    """Create a sub-chunk from a subset of article lines."""
    text = "\n".join(line.text for line in lines)
    highlights = [line.to_highlight() for line in lines]
    
    # Derive doc_id from source filename
    doc_id = re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)
    
    return ArticleChunk(
        text=text,
        source=source,
        doc_id=doc_id,
        article_id=article.article_id,
        article_number=article.article_number,
        section_path=article.section_path,
        page_start=lines[0].page if lines else 0,
        page_end=lines[-1].page if lines else 0,
        highlights=highlights,
        sub_chunk_index=sub_index,
        total_sub_chunks=total_subs,
        category=category,
    )


def chunk_articles(
    document: ExtractedDocument,
    max_tokens: int = 1500,
) -> list[ArticleChunk]:
    """
    Chunk a document into article-based chunks.
    
    Each article becomes one chunk. Long articles are split into
    sub-chunks at paragraph boundaries.
    
    Args:
        document: Extracted document with layout
        max_tokens: Maximum tokens per chunk (for sub-chunking)
        
    Returns:
        List of ArticleChunk objects
    """
    encoding = _get_encoding()
    chunks: list[ArticleChunk] = []
    
    # Detect document structure
    articles = detect_structure(document)
    
    for article in articles:
        if not article.lines:
            continue
        
        article_text = article.text
        article_tokens = _count_tokens(article_text, encoding)
        
        if article_tokens <= max_tokens:
            # Single chunk for this article
            doc_id = re.sub(r"\.pdf$", "", document.source, flags=re.IGNORECASE)
            
            chunk = ArticleChunk(
                text=article_text,
                source=document.source,
                doc_id=doc_id,
                article_id=article.article_id,
                article_number=article.article_number,
                section_path=article.section_path,
                page_start=article.page_start,
                page_end=article.page_end,
                highlights=article.highlights,
                sub_chunk_index=0,
                total_sub_chunks=1,
                category=document.category,
            )
            chunks.append(chunk)
        else:
            # Split into sub-chunks
            paragraphs = _split_article_into_paragraphs(article)
            
            if not paragraphs:
                continue
            
            # Group paragraphs into sub-chunks
            sub_chunks_lines: list[list[LineBlock]] = []
            current_lines: list[LineBlock] = []
            current_tokens = 0
            
            for para_lines in paragraphs:
                para_text = "\n".join(line.text for line in para_lines)
                para_tokens = _count_tokens(para_text, encoding)
                
                if current_tokens + para_tokens > max_tokens and current_lines:
                    # Save current sub-chunk and start new one
                    sub_chunks_lines.append(current_lines)
                    current_lines = para_lines.copy()
                    current_tokens = para_tokens
                else:
                    current_lines.extend(para_lines)
                    current_tokens += para_tokens
            
            # Don't forget remaining lines
            if current_lines:
                sub_chunks_lines.append(current_lines)
            
            # Create ArticleChunk objects for each sub-chunk
            total_subs = len(sub_chunks_lines)
            for i, sub_lines in enumerate(sub_chunks_lines):
                chunk = _create_sub_chunk(
                    article=article,
                    lines=sub_lines,
                    sub_index=i,
                    total_subs=total_subs,
                    source=document.source,
                    category=document.category,
                )
                chunks.append(chunk)
    
    return chunks


# ==============================================================================
# Legacy Chunking (for contracts)
# ==============================================================================


def _extract_section_header(text: str) -> str | None:
    """
    Try to extract a section header from the start of text.
    """
    lines = text.strip().split("\n")
    if not lines:
        return None

    first_line = lines[0].strip()

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

    if first_line.isupper() and len(first_line) < 100:
        return first_line

    return None


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _get_overlap_text(text: str, overlap_tokens: int, encoding: tiktoken.Encoding) -> str:
    """Get the last N tokens of text for overlap."""
    tokens = encoding.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    overlap_token_ids = tokens[-overlap_tokens:]
    return encoding.decode(overlap_token_ids)


def chunk_document(
    document: DocumentContent,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Chunk a document with smart boundaries and metadata (legacy format).
    
    Kept for backward compatibility with contract ingestion.
    
    Args:
        document: The extracted document content
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of Chunk objects with metadata
    """
    encoding = _get_encoding()
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        paragraphs = _split_into_paragraphs(page.text)
        current_chunk_text = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = _count_tokens(para, encoding)

            if para_tokens > chunk_size:
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
                                section_header=_extract_section_header(current_chunk_text),
                                category=document.category,
                            )
                        )
                        chunk_index += 1
                        overlap_text = _get_overlap_text(
                            current_chunk_text, chunk_overlap, encoding
                        )
                        current_chunk_text = overlap_text + " " + sentence
                        current_tokens = _count_tokens(current_chunk_text, encoding)
                    else:
                        current_chunk_text += " " + sentence
                        current_tokens += sent_tokens

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
                overlap_text = _get_overlap_text(current_chunk_text, chunk_overlap, encoding)
                current_chunk_text = overlap_text + "\n\n" + para
                current_tokens = _count_tokens(current_chunk_text, encoding)
            else:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
                current_tokens += para_tokens

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
