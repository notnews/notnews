## LLM-based Classification

### llm_classify_news

Classify news articles using Large Language Models (Claude, OpenAI) with flexible custom categories and web content fetching.

- **Arguments:**
  - `df`: pandas dataframe. No default.
  - `col`: column with the story text. Default is `text`
  - `provider`: LLM provider (`"claude"` or `"openai"`). Default is `"claude"`
  - `categories`: optional dictionary of custom categories. Default uses standard categories.
  - `api_key`: optional API key (uses environment variable if not provided)
  - `fetch_content`: whether to fetch full content from URLs. Default is `False`
  - `url_col`: column with URLs if fetch_content=True. Default is `"url"`

- **Functionality:**
  - Classifies articles using state-of-the-art language models
  - Provides confidence scores and reasoning for each classification
  - Supports custom category definitions with descriptions and examples
  - Can fetch and analyze full web content from URLs
  - Handles rate limiting and error recovery automatically

- **Output:**
  - Appends columns with classification results:
    - `llm_category_{provider}`: predicted category name
    - `llm_confidence_{provider}`: confidence score (0-1) 
    - `llm_reasoning_{provider}`: brief explanation of classification

- **Setup:**
  
  Install with LLM dependencies:
  ```bash
  pip install notnews[llm]
  # or
  uv add notnews --extra llm
  ```
  
  Set API keys:
  ```bash
  export ANTHROPIC_API_KEY="your_claude_api_key"
  export OPENAI_API_KEY="your_openai_api_key"
  ```

- **Examples:**

  **Basic Classification:**
  ```python
  >>> import pandas as pd
  >>> from notnews import llm_classify_news
  >>> 
  >>> df = pd.DataFrame({
  ...     'text': [
  ...         'The Federal Reserve raised interest rates by 0.25%',
  ...         'Taylor Swift spotted at Kansas City Chiefs game'
  ...     ]
  ... })
  >>> 
  >>> result = llm_classify_news(df, provider='claude')
  >>> print(result[['text', 'llm_category_claude', 'llm_confidence_claude']])
                                            text llm_category_claude  llm_confidence_claude
  0  The Federal Reserve raised interest rates by 0.25%           hard_news                   0.95
  1   Taylor Swift spotted at Kansas City Chiefs game           soft_news                   0.92
  ```

  **Custom Categories:**
  ```python
  >>> custom_categories = {
  ...     'finance': {
  ...         'description': 'Financial news, markets, economic policy',
  ...         'examples': ['Stock market updates', 'Federal Reserve decisions']
  ...     },
  ...     'entertainment': {
  ...         'description': 'Celebrity news, movies, music, entertainment',
  ...         'examples': ['Celebrity sightings', 'Movie releases']
  ...     }
  ... }
  >>> 
  >>> result = llm_classify_news(df, provider='claude', categories=custom_categories)
  >>> print(result[['text', 'llm_category_claude']])
                                            text llm_category_claude
  0  The Federal Reserve raised interest rates by 0.25%             finance
  1   Taylor Swift spotted at Kansas City Chiefs game       entertainment
  ```

  **Web Content Fetching:**
  ```python
  >>> df_with_urls = pd.DataFrame({
  ...     'text': ['Breaking economic news', 'Celebrity update'],
  ...     'url': ['https://example.com/economy', 'https://example.com/celeb']
  ... })
  >>> 
  >>> result = llm_classify_news(df_with_urls, 
  ...                           provider='claude', 
  ...                           fetch_content=True, 
  ...                           url_col='url')
  >>> print(result[['url', 'llm_category_claude', 'llm_fetched_content']])
  ```

- **Default Categories:**
  - `hard_news`: Politics, economics, international affairs, policy
  - `soft_news`: Entertainment, lifestyle, sports, celebrity coverage  
  - `opinion`: Editorial content, opinion pieces, analysis
  - `business`: Business news, corporate announcements, finance
  - `technology`: Tech industry, product launches, innovation
  - `science`: Scientific discoveries, research, health studies