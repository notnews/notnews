#!/usr/bin/env python

"""
Utilities for notnews package.

Consolidated module providing text processing, web content fetching, and other
utility functions.
"""

import logging
import re
import string
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

logger = logging.getLogger(__name__)

_PUNCTUATION_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

# Web scraping headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# Text processing functions
def clean_text(text: str) -> str:
    """Clean and normalize text for machine learning processing.

    Performs deterministic tokenization and normalization.

    Args:
        text: Input text to clean and normalize.

    Returns:
        Normalized, whitespace-separated text.

    Example:
        >>> import notnews
        >>> clean = notnews.clean_text("The politician announced new policies today!")
        >>> print(clean)
        the politician announced new policies today
    """
    normalized = re.sub(r"\d+", "", str(text or "").lower())
    tokens = normalized.translate(_PUNCTUATION_TABLE).split()
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """
    Tokenize and stem text.

    Args:
        text: Input text to tokenize

    Returns:
        List of stemmed tokens
    """
    return text.lower().translate(_PUNCTUATION_TABLE).split()


def truncate_text(text: str, max_tokens: int = 3000) -> str:
    """
    Truncate text to fit within token limits.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens (approximate)

    Returns:
        Truncated text
    """
    # Rough estimation: 1 token ≈ 4 characters
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    # Try to truncate at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    last_newline = truncated.rfind("\n")

    # Use whichever boundary is later
    boundary = max(last_period, last_newline)
    if boundary > max_chars * 0.8:  # If boundary in last 20%
        truncated = truncated[: boundary + 1]

    return truncated + "..."


# Web content fetching functions
def fetch_web_content(url: str, timeout: int = 10) -> str | None:
    """Fetch and extract clean text content from a web page.

    Downloads the web page, parses HTML, and extracts the main article content
    using common content selectors. Automatically cleans extracted text.

    Args:
        url: URL to fetch content from. Must include scheme (http/https).
        timeout: Request timeout in seconds. Defaults to 10.

    Returns:
        Extracted and cleaned text content, or None if fetching fails or
        content is too short (< 100 characters).

    Example:
        >>> import notnews
        >>> content = notnews.fetch_web_content("https://example.com")
        >>> if content:
        ...     print(f"Extracted {len(content)} characters")
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Invalid URL: {url}")
            return None

        # Fetch the page
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        response.raise_for_status()

        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Try to find main content areas
        main_content = None

        # Common article content selectors
        content_selectors = [
            "article",
            "main",
            '[role="main"]',
            ".article-content",
            ".post-content",
            ".entry-content",
            "#content",
            ".content",
            ".story-body",
            ".article-body",
        ]

        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break

        # If no main content found, use body
        if not main_content:
            main_content = soup.body if soup.body else soup

        # Extract text
        text = main_content.get_text(separator=" ", strip=True)

        # Clean up text
        text = _clean_web_content(text)

        # Check if we got meaningful content
        if len(text) < 100:
            logger.warning(f"Extracted text too short from {url}")
            return None

        return text

    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {e}")
        return None


def _clean_web_content(text: str) -> str:
    """Clean extracted web content."""
    # Remove excessive whitespace
    lines = text.split("\n")
    lines = [" ".join(line.split()) for line in lines]
    lines = [line for line in lines if line.strip()]
    text = "\n".join(lines)

    # Remove very short lines that are likely navigation/ads
    lines = text.split("\n")
    meaningful_lines = []
    for line in lines:
        # Keep lines that are likely content (more than 20 chars)
        if len(line) > 20:
            meaningful_lines.append(line)

    if meaningful_lines:
        text = "\n".join(meaningful_lines)

    # Limit text length to avoid token limits
    max_chars = 10000  # Approximately 2500 tokens
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text.strip()


def _tag_attribute(tag: Tag | None, name: str) -> str | None:
    """Return a scalar HTML attribute.

    Args:
        tag: Parsed HTML tag.
        name: Attribute name.

    Returns:
        String value, or None when absent or multi-valued.
    """
    if tag is None:
        return None
    value = tag.get(name)
    return value if isinstance(value, str) else None


def extract_article_metadata(
    url: str, html_content: str | bytes | None = None
) -> dict[str, Any]:
    """
    Extract metadata from an article page.

    Args:
        url: URL of the article
        html_content: Optional pre-fetched HTML content

    Returns:
        Dictionary containing title, author, date, and other metadata
    """
    metadata: dict[str, Any] = {"url": url}

    try:
        if not html_content:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
            html_content = response.content

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title_tag = soup.find("meta", property="og:title")
        title = _tag_attribute(
            title_tag if isinstance(title_tag, Tag) else None, "content"
        )
        if title is None and soup.title:
            title = str(soup.title.string) if soup.title.string is not None else None
        metadata["title"] = title

        # Extract description
        description_tag = soup.find("meta", property="og:description")
        description = _tag_attribute(
            description_tag if isinstance(description_tag, Tag) else None, "content"
        )
        if description is None:
            fallback_description = soup.find("meta", attrs={"name": "description"})
            description = _tag_attribute(
                fallback_description if isinstance(fallback_description, Tag) else None,
                "content",
            )
        metadata["description"] = description

        # Extract author
        author_tag = soup.find("meta", attrs={"name": "author"})
        author = _tag_attribute(
            author_tag if isinstance(author_tag, Tag) else None, "content"
        )
        if author is None:
            fallback_author = soup.find("meta", property="article:author")
            author = _tag_attribute(
                fallback_author if isinstance(fallback_author, Tag) else None,
                "content",
            )
        metadata["author"] = author

        # Extract publication date
        published_tag = soup.find("meta", property="article:published_time")
        pub_date = _tag_attribute(
            published_tag if isinstance(published_tag, Tag) else None, "content"
        )
        if pub_date is None:
            time_element = soup.find("time")
            if isinstance(time_element, Tag):
                pub_date = _tag_attribute(time_element, "datetime")
                if pub_date is None and time_element.string is not None:
                    pub_date = str(time_element.string)
        metadata["published_date"] = pub_date

    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")

    return metadata
