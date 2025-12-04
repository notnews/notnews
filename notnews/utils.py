#!/usr/bin/env python

"""
Utilities for notnews package.

Consolidated module providing text processing, web content fetching,
model downloading, and other utility functions.
"""

import logging
import os
import re
import string
from os import path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional imports with graceful fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # Download required NLTK data
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    _stemmer = PorterStemmer()
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    _stemmer = None

logger = logging.getLogger(__name__)

# Model repository URL
REPO_BASE_URL = (
    os.environ.get("NOTNEWS_DATA_URL")
    or "https://github.com/notnews/notnews/raw/master/notnews/"
)

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

    Performs tokenization, stemming, stopword removal, and normalization.
    Falls back to basic cleaning if NLTK is not available.

    Args:
        text: Input text to clean and normalize.

    Returns:
        Cleaned text with stemming, stopword removal, and normalization applied.
        Returns original text if cleaning fails.

    Example:
        >>> import notnews
        >>> clean = notnews.clean_text("The politician announced new policies today!")
        >>> print(clean)
        politician announc new polici today
    """
    if not HAS_NLTK:
        logger.warning("NLTK not available, returning basic cleaned text")
        # Basic cleaning without NLTK
        text = re.sub(r"\d+", "", str(text).lower())
        return "".join([ch for ch in text if ch not in string.punctuation])

    try:
        text = str(text) if text is not None else ""
        text = re.sub(r"\d+", "", text)
        text = text.lower()
        text = "".join([ch for ch in text if ch not in string.punctuation])

        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords.words("english")]
        stems = [_stemmer.stem(token) for token in tokens]

        return " ".join(stems)

    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return str(text) if text is not None else ""


def tokenize(text: str) -> list[str]:
    """
    Tokenize and stem text.

    Args:
        text: Input text to tokenize

    Returns:
        List of stemmed tokens
    """
    if not HAS_NLTK:
        # Basic tokenization without NLTK
        text = "".join([ch for ch in text if ch not in string.punctuation])
        return text.lower().split()

    try:
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        return [_stemmer.stem(token) for token in tokens]
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return text.lower().split()


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
        >>> content = notnews.fetch_web_content("https://example.com/article")
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


def extract_article_metadata(url: str, html_content: str = None) -> dict[str, Any]:
    """
    Extract metadata from an article page.

    Args:
        url: URL of the article
        html_content: Optional pre-fetched HTML content

    Returns:
        Dictionary containing title, author, date, and other metadata
    """
    metadata = {"url": url}

    try:
        if not html_content:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
            html_content = response.content

        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = None
        if soup.find("meta", property="og:title"):
            title = soup.find("meta", property="og:title").get("content")
        elif soup.title:
            title = soup.title.string
        metadata["title"] = title

        # Extract description
        description = None
        if soup.find("meta", property="og:description"):
            description = soup.find("meta", property="og:description").get("content")
        elif soup.find("meta", attrs={"name": "description"}):
            description = soup.find("meta", attrs={"name": "description"}).get(
                "content"
            )
        metadata["description"] = description

        # Extract author
        author = None
        if soup.find("meta", attrs={"name": "author"}):
            author = soup.find("meta", attrs={"name": "author"}).get("content")
        elif soup.find("meta", property="article:author"):
            author = soup.find("meta", property="article:author").get("content")
        metadata["author"] = author

        # Extract publication date
        pub_date = None
        if soup.find("meta", property="article:published_time"):
            pub_date = soup.find("meta", property="article:published_time").get(
                "content"
            )
        elif soup.find("time"):
            time_elem = soup.find("time")
            pub_date = time_elem.get("datetime") or time_elem.string
        metadata["published_date"] = pub_date

    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")

    return metadata


# Model downloading functions
def get_app_file_path(app_name: str, filename: str) -> str:
    """Get path for application data file."""
    user_dir = path.expanduser("~")
    app_data_dir = path.join(user_dir, "." + app_name)
    if not path.exists(app_data_dir):
        os.makedirs(app_data_dir)
    return path.join(app_data_dir, filename)


def download_file(url: str, target: str) -> bool:
    """
    Download file from URL to target path with progress bar.

    Args:
        url: URL to download from
        target: Local path to save file

    Returns:
        True if successful, False otherwise
    """
    # Create target directory if needed
    os.makedirs(os.path.dirname(target), exist_ok=True)

    # Set up headers for authentication if available
    headers = {}
    if "NOTNEWS_AUTH_TOKEN" in os.environ:
        auth_token = os.environ["NOTNEWS_AUTH_TOKEN"]
        headers["Authorization"] = f"token {auth_token}"

    try:
        # Streaming download with progress bar
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()

        chunk_size = 64 * 1024
        total_size = int(response.headers.get("content-length", 0))

        with open(target, "wb") as f:
            if total_size > 0:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {os.path.basename(target)}",
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # No content length, download without progress
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Downloaded {url} to {target}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


# Legacy utility functions for compatibility
def find_ngrams(vocab: list, text: str, n: int) -> list[int]:
    """
    Find and return list of the index of n-grams in the vocabulary list.

    Args:
        vocab: Vocabulary list
        text: Input text
        n: N-grams size

    Returns:
        List of indices of n-grams in vocabulary
    """
    wi = []

    ngram_iter = zip(*[text[i:] for i in range(n)], strict=False)
    for ngram in ngram_iter:
        word = "".join(ngram)
        try:
            idx = vocab.index(word)
        except (ValueError, IndexError):
            idx = 0
        wi.append(idx)

    return wi
