## Command Line

We also implement the scripts to process the input file in the CSV format:

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