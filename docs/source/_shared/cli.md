## Command Line

We provide scripts to process input files in CSV format for both traditional ML and LLM-based classification:

### LLM-based Classification

#### llm_classify_news

```
usage: llm_classify_news [-h] [-o OUTPUT] [-t TEXT] [-p {claude,openai}] 
                         [-f] [-u URL] input

Classify news articles using Large Language Models (Claude/OpenAI)

positional arguments:
input                 Input CSV file containing articles to classify

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with classification results 
                        (default: llm-news-classification-output.csv)
-t TEXT, --text TEXT  Name of column containing text to classify (default: text)
-p {claude,openai}, --provider {claude,openai}
                        LLM provider to use (default: claude)
-f, --fetch-content   Fetch full content from URLs before classification
-u URL, --url URL     Column name containing URLs if fetch-content is enabled 
                        (default: url)
```

**Note:** Requires API keys set as environment variables:
- `ANTHROPIC_API_KEY` for Claude
- `OPENAI_API_KEY` for OpenAI

### Traditional ML Classification

### 1. soft_news_url_cat_us

```
usage: soft_news_url_cat_us [-h] [-o OUTPUT] [-u URL] input

US Soft News Category by URL pattern

positional arguments:
input                 Input file

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with category data
-u URL, --url URL     Name or index location of column contains the domain
                        or URL (default: url)
```

### 2. pred_soft_news_us

```
usage: pred_soft_news_us [-h] [-o OUTPUT] [-t TEXT] input

Predict Soft News by text using NYT Soft News model

positional arguments:
input                 Input file

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with prediction data
-t TEXT, --text TEXT  Name or index location of column contains the text
                        (default: text)
```

### 3. pred_what_news_us

```
usage: pred_what_news_us [-h] [-o OUTPUT] [-t TEXT] input

Predict What News by text using NYT What News model

positional arguments:
input                 Input file

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with prediction data
-t TEXT, --text TEXT  Name or index location of column contains the text
                        (default: text)
```

### 4. soft_news_url_cat_uk

```
usage: soft_news_url_cat_uk [-h] [-o OUTPUT] [-u URL] input

UK Soft News Category by URL pattern

positional arguments:
input                 Input file

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with category data
-u URL, --url URL     Name or index location of column contains the domain
                        or URL (default: url)
```

### 5. pred_soft_news_uk

```
usage: pred_soft_news_uk [-h] [-o OUTPUT] [-t TEXT] input

Predict Soft News by text using UK URL Soft News model

positional arguments:
input                 Input file

optional arguments:
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                        Output file with prediction data
-t TEXT, --text TEXT  Name or index location of column contains the text
                        (default: text)
```