"""Version-independent inference for the published linear classifiers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from itertools import islice
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ._resources import ASSET_FILES, resolve_asset
from ._schemas import ASSET_SCHEMAS

_INFERENCE_BATCH_SIZE = 4096


@dataclass(frozen=True)
class _Estimator:
    """One fitted classifier and its one-vs-rest calibration curves."""

    coefficients: np.ndarray
    intercepts: np.ndarray
    x_thresholds: tuple[np.ndarray, ...]
    y_thresholds: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PortableTextClassifier:
    """A text vectorizer and calibrated linear classifier."""

    vocabulary: dict[str, int]
    ngram_range: tuple[int, int]
    classes: tuple[str, ...]
    estimators: tuple[_Estimator, ...]

    def transform(self, documents: Iterable[str]) -> np.ndarray:
        """Convert documents to the count matrix used during training."""
        values = list(documents)
        features = np.zeros((len(values), len(self.vocabulary)), dtype=np.float64)
        minimum, maximum = self.ngram_range

        for row, document in enumerate(values):
            tokens = re.sub(r"\d+", "[NUM]", str(document).lower()).split()
            for width in range(minimum, maximum + 1):
                for start in range(len(tokens) - width + 1):
                    feature = " ".join(tokens[start : start + width])
                    feature_index = self.vocabulary.get(feature)
                    if feature_index is not None:
                        features[row, feature_index] += 1

        return features

    def predict_proba(self, documents: Iterable[str]) -> np.ndarray:
        """Return calibrated class probabilities for each document."""
        iterator = iter(documents)
        batches = []
        while values := list(islice(iterator, _INFERENCE_BATCH_SIZE)):
            batches.append(self._predict_feature_batch(self.transform(values)))

        if not batches:
            return np.empty((0, len(self.classes)), dtype=np.float64)
        return np.concatenate(batches)

    def _predict_feature_batch(self, features: np.ndarray) -> np.ndarray:
        """Score one bounded feature batch."""
        mean_probability = np.zeros(
            (features.shape[0], len(self.classes)), dtype=np.float64
        )

        for estimator in self.estimators:
            scores = np.dot(features, estimator.coefficients.T) + estimator.intercepts
            probabilities = np.zeros_like(mean_probability)
            for output_index, (score, x_thresholds, y_thresholds) in enumerate(
                zip(
                    scores.T,
                    estimator.x_thresholds,
                    estimator.y_thresholds,
                    strict=True,
                )
            ):
                calibrated = np.interp(score, x_thresholds, y_thresholds)
                target_index = 1 if len(self.classes) == 2 else output_index
                probabilities[:, target_index] = calibrated

            if len(self.classes) == 2:
                probabilities[:, 0] = 1 - probabilities[:, 1]
            else:
                denominator = probabilities.sum(axis=1, keepdims=True)
                probabilities = np.divide(
                    probabilities,
                    denominator,
                    out=np.full_like(probabilities, 1 / len(self.classes)),
                    where=denominator != 0,
                )

            mean_probability += probabilities

        return mean_probability / len(self.estimators)

    def predict(self, documents: Iterable[str]) -> np.ndarray:
        """Return the highest-probability class for each document."""
        probabilities = self.predict_proba(documents)
        labels = np.asarray(self.classes, dtype=str)
        return labels[np.argmax(probabilities, axis=1)]


@cache
def _asset_rows(filename: str) -> list[dict[str, Any]]:
    """Read one validated Parquet asset into Python rows."""
    if filename not in ASSET_FILES:
        raise ValueError(f"Unknown model asset: {filename}")
    table = pq.read_table(resolve_asset(filename))
    if table.schema != ASSET_SCHEMAS[filename]:
        raise ValueError(f"Unexpected schema for {filename}")
    return table.to_pylist()


@cache
def load_model(model_name: str) -> PortableTextClassifier:
    """Load one classifier from the shared typed assets."""
    metadata = [
        row for row in _asset_rows("metadata.parquet") if row["model"] == model_name
    ]
    if len(metadata) != 1:
        raise ValueError(f"Expected one metadata row for {model_name}")

    vocabulary_rows = sorted(
        (
            row
            for row in _asset_rows("vocabulary.parquet")
            if row["model"] == model_name
        ),
        key=lambda row: row["feature_index"],
    )
    expected_indices = list(range(len(vocabulary_rows)))
    if [row["feature_index"] for row in vocabulary_rows] != expected_indices:
        raise ValueError(f"Vocabulary indices are not contiguous for {model_name}")
    vocabulary = {row["feature"]: row["feature_index"] for row in vocabulary_rows}

    estimator_rows = [
        row for row in _asset_rows("estimators.parquet") if row["model"] == model_name
    ]
    calibrator_rows = [
        row for row in _asset_rows("calibrators.parquet") if row["model"] == model_name
    ]
    folds = sorted({row["fold"] for row in estimator_rows})
    estimators = []
    for fold in folds:
        fold_estimators = sorted(
            (row for row in estimator_rows if row["fold"] == fold),
            key=lambda row: row["output_index"],
        )
        fold_calibrators = sorted(
            (row for row in calibrator_rows if row["fold"] == fold),
            key=lambda row: row["output_index"],
        )
        output_indices = list(range(len(fold_estimators)))
        if [row["output_index"] for row in fold_estimators] != output_indices:
            raise ValueError(f"Estimator outputs are not contiguous for {model_name}")
        if [row["output_index"] for row in fold_calibrators] != output_indices:
            raise ValueError(f"Calibrator outputs do not match for {model_name}")

        coefficients = np.asarray(
            [row["coefficients"] for row in fold_estimators], dtype=np.float64
        )
        if coefficients.shape[1] != len(vocabulary):
            raise ValueError(f"Estimator feature count differs for {model_name}")
        x_thresholds = tuple(
            np.asarray(row["x_thresholds"], dtype=np.float64)
            for row in fold_calibrators
        )
        y_thresholds = tuple(
            np.asarray(row["y_thresholds"], dtype=np.float64)
            for row in fold_calibrators
        )
        if any(
            x.shape != y.shape for x, y in zip(x_thresholds, y_thresholds, strict=True)
        ):
            raise ValueError(f"Calibration thresholds differ for {model_name}")

        estimators.append(
            _Estimator(
                coefficients=coefficients,
                intercepts=np.asarray(
                    [row["intercept"] for row in fold_estimators], dtype=np.float64
                ),
                x_thresholds=x_thresholds,
                y_thresholds=y_thresholds,
            )
        )

    if not estimators:
        raise ValueError(f"No estimators found for {model_name}")

    row = metadata[0]
    return PortableTextClassifier(
        vocabulary=vocabulary,
        ngram_range=(row["ngram_min"], row["ngram_max"]),
        classes=tuple(row["class_labels"]),
        estimators=tuple(estimators),
    )
