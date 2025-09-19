#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenAI-based news classification implementation for notnews.

This module implements the OpenAINewsClassifier using OpenAI's GPT models
for news article classification.
"""

import os
import json
import logging
from typing import Dict, Optional, Any

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .llm_classifier import LLMNewsClassifier
from .llm_utils import prepare_content_for_llm, truncate_for_token_limit

# Configure logging
logger = logging.getLogger(__name__)


class OpenAINewsClassifier(LLMNewsClassifier):
    """
    OpenAI-based implementation for news classification.
    
    Uses OpenAI's GPT models to classify news articles into categories
    with confidence scores and reasoning.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI classifier.
        
        Args:
            api_key: Optional OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: OpenAI model to use (default: gpt-3.5-turbo for cost efficiency)
                   Other options: gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo-1106
        """
        super().__init__(api_key=api_key)
        self.provider = "openai"
        self.model = model
        
        if not HAS_OPENAI:
            logger.error("openai package not installed. Install with: pip install openai")
            self.client = None
        else:
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
            else:
                self.client = None
                logger.warning("No API key provided for OpenAI. Set OPENAI_API_KEY environment variable.")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        return os.getenv("OPENAI_API_KEY")
    
    def validate_api_key(self) -> bool:
        """Validate that API key is present and client is initialized."""
        if not HAS_OPENAI:
            logger.error("openai package not installed. Cannot use OpenAI classifier.")
            return False
        
        if not self.api_key:
            logger.error("No API key provided for OpenAI. Set OPENAI_API_KEY environment variable.")
            return False
        
        if not self.client:
            # Try to initialize client if we have a key
            if self.api_key:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    return False
        
        return True
    
    def _classify_text(self, text: str, categories: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Classify a single text using OpenAI API.
        
        Args:
            text: Text to classify
            categories: Dictionary of categories with descriptions
            
        Returns:
            Dictionary with category, confidence, and reasoning
        """
        if not self.validate_api_key():
            raise Exception("Invalid or missing API key for OpenAI")
        
        # Format the prompt
        categories_str = self.format_categories_for_prompt(categories)
        
        # Truncate text if needed to stay within token limits
        text = truncate_for_token_limit(text, max_tokens=3000)
        
        prompt = self.prompt_template.format(
            categories=categories_str,
            content=text
        )
        
        try:
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a news categorization expert. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=500,
                response_format={"type": "json_object"} if "gpt-4" in self.model or "3.5-turbo-1106" in self.model else None
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Clean up response if needed
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                result = json.loads(response_text.strip())
                
                # Validate response structure
                if not all(k in result for k in ["category", "confidence", "reasoning"]):
                    raise ValueError("Invalid response structure")
                
                # Ensure category is valid
                if result["category"] not in categories:
                    logger.warning(f"OpenAI returned invalid category: {result['category']}")
                    # Try to find the closest match or default to first category
                    result["category"] = list(categories.keys())[0]
                
                # Ensure confidence is a float between 0 and 1
                result["confidence"] = float(result["confidence"])
                if not 0 <= result["confidence"] <= 1:
                    result["confidence"] = min(max(result["confidence"], 0), 1)
                
                return result
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse OpenAI response: {e}")
                logger.debug(f"Raw response: {response_text}")
                
                # Fallback: try to extract information from text
                category = None
                for cat in categories.keys():
                    if cat.lower() in response_text.lower():
                        category = cat
                        break
                
                if not category:
                    category = list(categories.keys())[0]
                
                return {
                    "category": category,
                    "confidence": 0.5,
                    "reasoning": "Failed to parse structured response, using fallback classification"
                }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            
            # Handle rate limiting specifically
            if "rate_limit" in str(e).lower() or "429" in str(e):
                self.handle_rate_limit()
                # Retry once after rate limit
                return self._classify_text(text, categories)
            
            raise


def create_openai_classifier(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> OpenAINewsClassifier:
    """
    Factory function to create an OpenAI news classifier.
    
    Args:
        api_key: Optional OpenAI API key
        model: OpenAI model to use
        
    Returns:
        Configured OpenAINewsClassifier instance
    """
    return OpenAINewsClassifier(api_key=api_key, model=model)