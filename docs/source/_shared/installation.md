## Installation

Installation is as easy as typing in:

```bash
pip install notnews
```

For faster installation using UV:

```bash
uv add notnews
```

### Optional Dependencies

For LLM-based classification with Claude and OpenAI:

```bash
pip install notnews[llm]
# or
uv add notnews --extra llm
```

### Requirements

- Python 3.11, 3.12, 3.13
- scikit-learn 1.3+ (models trained with sklearn 0.22+ are automatically compatible)
- pandas, numpy, nltk, and other standard scientific Python packages

### LLM Requirements (Optional)

- anthropic>=0.18.0 (for Claude)
- openai>=1.12.0 (for OpenAI)

### Compatibility

This package includes automatic compatibility layers to ensure models trained with older scikit-learn versions (0.22+) work seamlessly with modern scikit-learn versions (1.3-1.5). Version warnings from scikit-learn are expected and harmless.