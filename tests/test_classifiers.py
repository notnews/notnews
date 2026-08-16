"""Offline tests for classifier data handling."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from notnews import classifiers


def test_predict_soft_news_preserves_index_and_skips_missing_text() -> None:
    """Predictions align to valid rows without classifying missing text."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    source = pd.DataFrame({"text": ["first", None, "third"]}, index=[8, 3, 5])

    with patch.object(classifiers, "_load_model", return_value=model):
        result = classifiers.predict_soft_news(source)

    assert result.index.tolist() == [8, 3, 5]
    assert result.loc[8, "prob_soft_news_us"] == pytest.approx(0.8)
    assert pd.isna(result.loc[3, "prob_soft_news_us"])
    assert result.loc[5, "prob_soft_news_us"] == pytest.approx(0.4)
    transformed = model.predict_proba.call_args.args[0]
    assert transformed.index.tolist() == [8, 5]
    assert transformed.tolist() == ["first", "third"]


def test_predict_soft_news_does_not_load_model_for_missing_text() -> None:
    """An all-missing input does not trigger an artifact download."""
    source = pd.DataFrame({"text": [None]}, index=[42])

    with patch.object(classifiers, "_load_model") as load_model_mock:
        result = classifiers.predict_soft_news(source)

    load_model_mock.assert_not_called()
    assert result.index.tolist() == [42]
    assert pd.isna(result.loc[42, "prob_soft_news_us"])


def test_predict_news_category_preserves_nondefault_index() -> None:
    """Category probabilities are aligned to the input rather than RangeIndex."""
    classes = ["Hard", *classifiers.SOFT_NEWS_CATEGORIES]
    probabilities = np.zeros((2, len(classes)))
    probabilities[0, classes.index("Arts")] = 0.75
    probabilities[1, classes.index("Hard")] = 0.75
    probabilities[1, classes.index("Books")] = 0.25
    model = SimpleNamespace(
        classes=classes,
        predict_proba=Mock(return_value=probabilities),
    )
    source = pd.DataFrame({"text": ["arts", None, "politics"]}, index=[101, 44, 205])

    with patch.object(classifiers, "load_model", return_value=model):
        result = classifiers.predict_news_category(source)

    assert result.index.tolist() == [101, 44, 205]
    assert result.loc[101, "pred_category"] == "Arts"
    assert result.loc[101, "prob_soft_news"] == pytest.approx(0.75)
    assert pd.isna(result.loc[44, "pred_category"])
    assert pd.isna(result.loc[44, "prob_soft_news"])
    assert result.loc[205, "pred_category"] == "Hard"
    assert result.loc[205, "prob_soft_news"] == pytest.approx(0.25)
    model.predict_proba.assert_called_once()


def test_load_model_maps_region_to_portable_model() -> None:
    """Region names map to the corresponding published model."""
    model = object()

    with patch.object(classifiers, "load_model", return_value=model) as load:
        assert classifiers._load_model("us") is model

    load.assert_called_once_with("us_soft")


def test_load_model_rejects_unknown_region() -> None:
    """Unsupported regions fail before any artifact lookup."""
    with pytest.raises(ValueError, match="Unsupported region"):
        classifiers._load_model("ca")


def test_classify_by_url_accepts_empty_dataframe() -> None:
    """Empty inputs retain their index and receive typed output columns."""
    source = pd.DataFrame({"url": pd.Series(dtype="string")})

    result = classifiers.classify_by_url(source)

    assert result.empty
    assert list(result.columns) == ["url", "hard_news", "soft_news"]
    assert str(result["hard_news"].dtype) == "Int64"
    assert str(result["soft_news"].dtype) == "Int64"
