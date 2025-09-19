#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for LLM-based news classification.

Provides web content fetching, text extraction, and content cleaning utilities.
"""

import logging
import requests
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request headers to avoid being blocked
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


def fetch_web_content(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch and extract text content from a web page.
    
    Args:
        url: URL to fetch content from
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content or None if fetching fails
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
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        main_content = None
        
        # Common article content selectors
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '#content',
            '.content',
            '.story-body',
            '.article-body'
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
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = clean_text_content(text)
        
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


def clean_text_content(text: str) -> str:
    """
    Clean and normalize extracted text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = text.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    lines = [line for line in lines if line.strip()]
    text = '\n'.join(lines)
    
    # Remove very short lines that are likely navigation/ads
    lines = text.split('\n')
    meaningful_lines = []
    for line in lines:
        # Keep lines that are likely content (more than 20 chars)
        if len(line) > 20:
            meaningful_lines.append(line)
    
    if meaningful_lines:
        text = '\n'.join(meaningful_lines)
    
    # Limit text length to avoid token limits in LLMs
    max_chars = 10000  # Approximately 2500 tokens
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text.strip()


def extract_article_metadata(url: str, html_content: str = None) -> Dict[str, Any]:
    """
    Extract metadata from an article page.
    
    Args:
        url: URL of the article
        html_content: Optional pre-fetched HTML content
        
    Returns:
        Dictionary containing title, author, date, and other metadata
    """
    metadata = {'url': url}
    
    try:
        if not html_content:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
            html_content = response.content
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = None
        if soup.find('meta', property='og:title'):
            title = soup.find('meta', property='og:title').get('content')
        elif soup.title:
            title = soup.title.string
        metadata['title'] = title
        
        # Extract description
        description = None
        if soup.find('meta', property='og:description'):
            description = soup.find('meta', property='og:description').get('content')
        elif soup.find('meta', attrs={'name': 'description'}):
            description = soup.find('meta', attrs={'name': 'description'}).get('content')
        metadata['description'] = description
        
        # Extract author
        author = None
        if soup.find('meta', attrs={'name': 'author'}):
            author = soup.find('meta', attrs={'name': 'author'}).get('content')
        elif soup.find('meta', property='article:author'):
            author = soup.find('meta', property='article:author').get('content')
        metadata['author'] = author
        
        # Extract publication date
        pub_date = None
        if soup.find('meta', property='article:published_time'):
            pub_date = soup.find('meta', property='article:published_time').get('content')
        elif soup.find('time'):
            time_elem = soup.find('time')
            pub_date = time_elem.get('datetime') or time_elem.string
        metadata['published_date'] = pub_date
        
        # Extract keywords/tags
        keywords = None
        if soup.find('meta', attrs={'name': 'keywords'}):
            keywords = soup.find('meta', attrs={'name': 'keywords'}).get('content')
        elif soup.find('meta', property='article:tag'):
            tags = soup.find_all('meta', property='article:tag')
            keywords = ', '.join([tag.get('content') for tag in tags])
        metadata['keywords'] = keywords
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {url}: {e}")
    
    return metadata


def prepare_content_for_llm(
    text: str,
    url: Optional[str] = None,
    fetch_full: bool = False,
    include_metadata: bool = True
) -> str:
    """
    Prepare content for LLM classification.
    
    Args:
        text: Original text content
        url: Optional URL to fetch full content from
        fetch_full: Whether to fetch full content from URL
        include_metadata: Whether to include metadata in the content
        
    Returns:
        Prepared content string for LLM processing
    """
    content_parts = []
    
    # If URL provided and fetch requested, try to get full content
    if url and fetch_full:
        fetched_content = fetch_web_content(url)
        if fetched_content:
            # Use fetched content as primary
            content_parts.append("=== Full Article Content ===")
            content_parts.append(fetched_content)
            
            # Add original text as supplementary if different
            if text and text.strip() != fetched_content.strip():
                content_parts.append("\n=== Original Excerpt ===")
                content_parts.append(text)
        else:
            # Fallback to original text
            content_parts.append(text)
            
        # Add metadata if requested
        if include_metadata:
            metadata = extract_article_metadata(url)
            if any(metadata.values()):
                content_parts.append("\n=== Article Metadata ===")
                for key, value in metadata.items():
                    if value and key != 'url':
                        content_parts.append(f"{key.title()}: {value}")
    else:
        # Just use the provided text
        content_parts.append(text)
    
    return '\n'.join(content_parts)


def truncate_for_token_limit(text: str, max_tokens: int = 3000) -> str:
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
    
    # Try to truncate at a sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    
    # Use whichever boundary is later
    boundary = max(last_period, last_newline)
    if boundary > max_chars * 0.8:  # If we found a boundary in the last 20%
        truncated = truncated[:boundary + 1]
    
    return truncated + "..."