"""Tests for portable Parquet model inference."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from notnews import _portable_model
from notnews._portable_model import PortableTextClassifier, _Estimator
from notnews._schemas import ASSET_SCHEMAS


def test_binary_inference_vectorizes_ngrams_and_numbers() -> None:
    """Binary inference applies the exported vocabulary and calibration."""
    estimator = _Estimator(
        coefficients=np.array([[1.0, 2.0, 3.0]]),
        intercepts=np.array([0.0]),
        x_thresholds=(np.array([-10.0, 10.0]),),
        y_thresholds=(np.array([0.0, 1.0]),),
    )
    model = PortableTextClassifier(
        vocabulary={"election": 0, "[NUM]": 1, "election [NUM]": 2},
        ngram_range=(1, 2),
        classes=("0", "1"),
        estimators=(estimator,),
    )

    features = model.transform(["Election 2026"])
    assert features.tolist() == [[1.0, 1.0, 1.0]]
    np.testing.assert_allclose(model.predict_proba(["Election 2026"]), [[0.2, 0.8]])
    assert model.predict(["Election 2026"]).tolist() == ["1"]


def test_multiclass_zero_calibration_uses_uniform_distribution() -> None:
    """All-zero calibration curves produce a valid uniform distribution."""
    estimator = _Estimator(
        coefficients=np.zeros((3, 1)),
        intercepts=np.zeros(3),
        x_thresholds=(np.array([-1.0, 1.0]),) * 3,
        y_thresholds=(np.array([0.0, 0.0]),) * 3,
    )
    model = PortableTextClassifier(
        vocabulary={"news": 0},
        ngram_range=(1, 1),
        classes=("a", "b", "c"),
        estimators=(estimator,),
    )

    assert model.predict_proba(["news"]).tolist() == [
        [pytest.approx(1 / 3), pytest.approx(1 / 3), pytest.approx(1 / 3)]
    ]


def test_inference_vectorizes_in_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction does not allocate one corpus-sized dense feature matrix."""
    estimator = _Estimator(
        coefficients=np.array([[1.0]]),
        intercepts=np.array([0.0]),
        x_thresholds=(np.array([-1.0, 1.0]),),
        y_thresholds=(np.array([0.0, 1.0]),),
    )
    model = PortableTextClassifier(
        vocabulary={"news": 0},
        ngram_range=(1, 1),
        classes=("0", "1"),
        estimators=(estimator,),
    )
    batch_sizes = []
    original_transform = PortableTextClassifier.transform

    def tracked_transform(
        classifier: PortableTextClassifier, documents: list[str]
    ) -> np.ndarray:
        batch_sizes.append(len(documents))
        return original_transform(classifier, documents)

    monkeypatch.setattr(_portable_model, "_INFERENCE_BATCH_SIZE", 2)
    monkeypatch.setattr(PortableTextClassifier, "transform", tracked_transform)

    probabilities = model.predict_proba("news" for _ in range(5))

    assert probabilities.shape == (5, 2)
    assert batch_sizes == [2, 2, 1]


def _write_asset(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a test asset using the production schema."""
    pq.write_table(pa.Table.from_pylist(rows, schema=ASSET_SCHEMAS[path.name]), path)


def test_load_model_reads_typed_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader reconstructs one model from the shared Parquet contract."""
    _write_asset(
        tmp_path / "metadata.parquet",
        [
            {
                "model": "tiny",
                "ngram_min": 1,
                "ngram_max": 1,
                "class_labels": ["0", "1"],
            }
        ],
    )
    _write_asset(
        tmp_path / "vocabulary.parquet",
        [{"model": "tiny", "feature": "news", "feature_index": 0}],
    )
    _write_asset(
        tmp_path / "estimators.parquet",
        [
            {
                "model": "tiny",
                "fold": 0,
                "output_index": 0,
                "intercept": 0.0,
                "coefficients": [1.0],
            }
        ],
    )
    _write_asset(
        tmp_path / "calibrators.parquet",
        [
            {
                "model": "tiny",
                "fold": 0,
                "output_index": 0,
                "x_thresholds": [-1.0, 1.0],
                "y_thresholds": [0.0, 1.0],
            }
        ],
    )
    monkeypatch.setattr(
        _portable_model, "resolve_asset", lambda filename: str(tmp_path / filename)
    )
    _portable_model._asset_rows.cache_clear()
    _portable_model.load_model.cache_clear()

    model = _portable_model.load_model("tiny")

    assert model.classes == ("0", "1")
    assert model.predict_proba(["news"]).tolist() == [[0.0, 1.0]]
    _portable_model._asset_rows.cache_clear()
    _portable_model.load_model.cache_clear()


def test_asset_reader_rejects_wrong_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema drift fails before inference can silently change."""
    path = tmp_path / "metadata.parquet"
    pq.write_table(pa.table({"model": ["tiny"]}), path)
    monkeypatch.setattr(_portable_model, "resolve_asset", lambda filename: str(path))
    _portable_model._asset_rows.cache_clear()

    with pytest.raises(ValueError, match="Unexpected schema"):
        _portable_model._asset_rows("metadata.parquet")

    _portable_model._asset_rows.cache_clear()


def test_sample_fixtures_have_explicit_string_schemas() -> None:
    """Regression fixtures preserve their declared columns and nullability."""
    expected = {
        "sample_us.parquet": pa.schema(
            [
                pa.field("src", pa.string(), nullable=False),
                pa.field("url", pa.string()),
                pa.field("text", pa.string(), nullable=False),
            ]
        ),
        "sample_uk.parquet": pa.schema(
            [
                pa.field("src_name", pa.string(), nullable=False),
                pa.field("url", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
            ]
        ),
    }

    for filename, schema in expected.items():
        assert pq.read_schema(Path("tests") / filename) == schema
