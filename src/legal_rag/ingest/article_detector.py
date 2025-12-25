"""Legal document structure detection.

Detects article headers and hierarchical headings in French legal documents.
Groups lines into articles with section path metadata.
"""

import re
from dataclasses import dataclass, field

from .pdf_extractor import ExtractedDocument, LineBlock


# ==============================================================================
# Heading Patterns
# ==============================================================================

# Article patterns (these mark the start of a new article)
ARTICLE_PATTERNS = [
    # "Article 1", "Article 1-1", "Article 1bis", "Article 1er"
    re.compile(r"^(Article)\s+(\d+(?:[-–]\d+)?(?:bis|ter|quater|quinquies|er)?)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE),
    # "Art. 1", "Art 1"
    re.compile(r"^(Art\.?)\s+(\d+(?:[-–]\d+)?(?:bis|ter|quater|quinquies|er)?)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE),
]

# Higher-level heading patterns (for section path, not article breaks)
HEADING_PATTERNS = [
    # Match in order of hierarchy (most general first)
    (re.compile(r"^(Livre)\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "livre"),
    (re.compile(r"^(Partie)\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "partie"),
    (re.compile(r"^(Titre)\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "titre"),
    (re.compile(r"^(Chapitre)\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "chapitre"),
    (re.compile(r"^(Section)\s+(\d+|[IVXLCDM]+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "section"),
    (re.compile(r"^(Sous-section)\s+(\d+|[IVXLCDM]+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "sous-section"),
    (re.compile(r"^(Paragraphe)\s+(\d+|[IVXLCDM]+|§\s*\d+)\s*[.:\-–]?\s*(.*)$", re.IGNORECASE), "paragraphe"),
]

# Hierarchy order for section path management
HIERARCHY_ORDER = ["livre", "partie", "titre", "chapitre", "section", "sous-section", "paragraphe"]


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class SectionHeading:
    """A hierarchical heading (Book/Title/Chapter/Section)."""
    
    level: str  # e.g., "titre", "chapitre"
    number: str  # e.g., "I", "2"
    title: str  # Optional title text
    
    def __str__(self) -> str:
        """Format as 'Level Number: Title' or 'Level Number'."""
        base = f"{self.level.capitalize()} {self.number}"
        if self.title:
            return f"{base}: {self.title}"
        return base


@dataclass
class Article:
    """A legal article with its content and metadata."""
    
    article_id: str  # e.g., "Article 1134", "Art. 1"
    article_number: str  # e.g., "1134", "1"
    title: str  # Optional title after article number
    section_path: list[str]  # Hierarchical context, e.g., ["Livre I", "Titre II", "Chapitre 1"]
    lines: list[LineBlock]  # All lines belonging to this article
    
    @property
    def page_start(self) -> int:
        """First page containing this article."""
        if not self.lines:
            return 0
        return self.lines[0].page
    
    @property
    def page_end(self) -> int:
        """Last page containing this article."""
        if not self.lines:
            return 0
        return self.lines[-1].page
    
    @property
    def text(self) -> str:
        """Full article text from all lines."""
        return "\n".join(line.text for line in self.lines)
    
    @property
    def highlights(self) -> list[dict]:
        """All highlight positions for this article."""
        return [line.to_highlight() for line in self.lines]
    
    def get_full_id(self) -> str:
        """Get full identifier including section path."""
        path = " > ".join(self.section_path) if self.section_path else ""
        if path:
            return f"{path} > {self.article_id}"
        return self.article_id


# ==============================================================================
# Detection Functions
# ==============================================================================


def _match_article(text: str) -> tuple[str, str, str] | None:
    """
    Check if text matches an article header pattern.
    
    Returns:
        Tuple of (full_id, number, title) if match, None otherwise
    """
    for pattern in ARTICLE_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            prefix = match.group(1)  # "Article" or "Art."
            number = match.group(2)  # "1134" or "1-1"
            title = match.group(3).strip() if match.group(3) else ""
            
            # Normalize prefix
            full_id = f"Article {number}"
            return (full_id, number, title)
    
    return None


def _match_heading(text: str) -> SectionHeading | None:
    """
    Check if text matches a section heading pattern.
    
    Returns:
        SectionHeading if match, None otherwise
    """
    for pattern, level in HEADING_PATTERNS:
        match = pattern.match(text.strip())
        if match:
            number = match.group(2)
            title = match.group(3).strip() if match.group(3) else ""
            return SectionHeading(level=level, number=number, title=title)
    
    return None


def _update_section_path(
    current_path: list[str],
    new_heading: SectionHeading,
) -> list[str]:
    """
    Update section path when a new heading is encountered.
    
    When we see a heading at level N, we:
    1. Keep all headings at levels < N
    2. Replace/add the heading at level N
    3. Remove all headings at levels > N
    """
    new_path = []
    new_level_index = HIERARCHY_ORDER.index(new_heading.level)
    
    # Keep headings at higher levels (smaller index)
    for existing in current_path:
        # Parse the existing heading to get its level
        for level in HIERARCHY_ORDER[:new_level_index]:
            if existing.lower().startswith(level):
                new_path.append(existing)
                break
    
    # Add the new heading
    new_path.append(str(new_heading))
    
    return new_path


def detect_structure(document: ExtractedDocument) -> list[Article]:
    """
    Detect articles and hierarchical structure in a legal document.
    
    Walks through all lines and:
    1. Tracks current section path (Book/Title/Chapter/Section)
    2. Detects article headers and groups following lines
    
    Args:
        document: Extracted document with lines
        
    Returns:
        List of Article objects with section path metadata
    """
    articles: list[Article] = []
    current_section_path: list[str] = []
    current_article: Article | None = None
    
    # Track preamble lines (before first article)
    preamble_lines: list[LineBlock] = []
    
    for line in document.all_lines:
        text = line.text.strip()
        
        if not text:
            continue
        
        # Check for article header
        article_match = _match_article(text)
        if article_match:
            # Save current article if exists
            if current_article:
                articles.append(current_article)
            
            # Start new article
            full_id, number, title = article_match
            current_article = Article(
                article_id=full_id,
                article_number=number,
                title=title,
                section_path=current_section_path.copy(),
                lines=[line],  # Include the header line
            )
            continue
        
        # Check for section heading
        heading_match = _match_heading(text)
        if heading_match:
            current_section_path = _update_section_path(current_section_path, heading_match)
            
            # If we're in an article, this heading might be part of it
            # (some articles span multiple sub-sections)
            # For now, we treat headings as article boundaries only if
            # they're at a high level (livre, titre, chapitre)
            if heading_match.level in ["livre", "partie", "titre", "chapitre"]:
                if current_article:
                    articles.append(current_article)
                    current_article = None
            continue
        
        # Regular content line
        if current_article:
            current_article.lines.append(line)
        else:
            preamble_lines.append(line)
    
    # Don't forget the last article
    if current_article:
        articles.append(current_article)
    
    # Create a preamble article if we have content before first article
    if preamble_lines:
        preamble = Article(
            article_id="Préambule",
            article_number="0",
            title="",
            section_path=[],
            lines=preamble_lines,
        )
        articles.insert(0, preamble)
    
    return articles


def detect_articles_only(document: ExtractedDocument) -> list[Article]:
    """
    Simpler detection that only finds articles, ignoring hierarchy.
    
    Useful for documents where section headings aren't well-structured.
    """
    articles: list[Article] = []
    current_article: Article | None = None
    preamble_lines: list[LineBlock] = []
    
    for line in document.all_lines:
        text = line.text.strip()
        
        if not text:
            continue
        
        article_match = _match_article(text)
        if article_match:
            if current_article:
                articles.append(current_article)
            
            full_id, number, title = article_match
            current_article = Article(
                article_id=full_id,
                article_number=number,
                title=title,
                section_path=[],
                lines=[line],
            )
        elif current_article:
            current_article.lines.append(line)
        else:
            preamble_lines.append(line)
    
    if current_article:
        articles.append(current_article)
    
    if preamble_lines:
        preamble = Article(
            article_id="Préambule",
            article_number="0",
            title="",
            section_path=[],
            lines=preamble_lines,
        )
        articles.insert(0, preamble)
    
    return articles
