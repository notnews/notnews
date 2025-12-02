## Installation

Installation is as easy as typing in:

```bash
pip install notnews
```

For faster installation using UV:

```bash
uv add notnews
```

### Requirements

- Python 3.11, 3.12, 3.13
- scikit-learn 1.3+ (models trained with sklearn 0.22+ are automatically compatible)
- pandas, numpy, nltk, and other standard scientific Python packages

### Compatibility

This package includes automatic compatibility layers to ensure models trained with older scikit-learn versions (0.22+) work seamlessly with modern scikit-learn versions (1.3-1.5). Version warnings from scikit-learn are expected and harmless.