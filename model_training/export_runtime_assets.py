"""Export trusted joblib models as version-independent Parquet tables."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from notnews._schemas import ASSET_SCHEMAS

MODEL_FILES = {
    "us_soft": (
        "us_model/nyt_us_soft_news_classifier.joblib",
        "us_model/nyt_us_soft_news_vectorizer.joblib",
    ),
    "uk_soft": (
        "uk_model/url_uk_classifier.joblib",
        "uk_model/url_uk_vectorizer.joblib",
    ),
    "us_category": (
        "us_model/nyt_us_classifier.joblib",
        "us_model/nyt_us_vectorizer.joblib",
    ),
}


def custom_tokenizer(document: str) -> list[str]:
    """Provide the symbol referenced by the trusted vectorizer pickles."""
    return re.sub(r"\d+", "[NUM]", document).split()


def _rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Extract portable rows from the trusted source artifacts."""
    sys.modules["__main__"].__dict__["custom_tokenizer"] = custom_tokenizer
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ASSET_SCHEMAS}

    for model_name, (classifier_file, vectorizer_file) in MODEL_FILES.items():
        classifier = joblib.load(source_dir / classifier_file)
        vectorizer = joblib.load(source_dir / vectorizer_file)
        vocabulary = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])

        if [index for _, index in vocabulary] != list(range(len(vocabulary))):
            raise ValueError(f"{model_name} vocabulary indices are not contiguous")
        if classifier.n_features_in_ != len(vocabulary):
            raise ValueError(f"{model_name} classifier and vocabulary sizes differ")

        rows["metadata.parquet"].append(
            {
                "model": model_name,
                "ngram_min": vectorizer.ngram_range[0],
                "ngram_max": vectorizer.ngram_range[1],
                "class_labels": [str(label) for label in classifier.classes_],
            }
        )
        rows["vocabulary.parquet"].extend(
            {
                "model": model_name,
                "feature": feature,
                "feature_index": index,
            }
            for feature, index in vocabulary
        )

        for fold, calibrated in enumerate(classifier.calibrated_classifiers_):
            coefficients = np.asarray(calibrated.estimator.coef_, dtype=np.float64)
            intercepts = np.asarray(calibrated.estimator.intercept_, dtype=np.float64)
            if coefficients.shape[1] != len(vocabulary):
                raise ValueError(
                    f"{model_name} fold {fold} has the wrong feature count"
                )
            if len(calibrated.calibrators) != coefficients.shape[0]:
                raise ValueError(f"{model_name} fold {fold} has unmatched calibrators")

            for output_index, (coefficient, intercept, calibrator) in enumerate(
                zip(coefficients, intercepts, calibrated.calibrators, strict=True)
            ):
                rows["estimators.parquet"].append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "output_index": output_index,
                        "intercept": float(intercept),
                        "coefficients": coefficient.tolist(),
                    }
                )
                x_thresholds = np.asarray(calibrator.X_thresholds_, dtype=np.float64)
                y_thresholds = np.asarray(calibrator.y_thresholds_, dtype=np.float64)
                if x_thresholds.shape != y_thresholds.shape:
                    raise ValueError(
                        f"{model_name} fold {fold} calibration sizes differ"
                    )
                rows["calibrators.parquet"].append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "output_index": output_index,
                        "x_thresholds": x_thresholds.tolist(),
                        "y_thresholds": y_thresholds.tolist(),
                    }
                )

    return rows


def export(source_dir: Path, output_dir: Path) -> None:
    """Write typed, compressed Parquet inference assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in _rows(source_dir).items():
        path = output_dir / filename
        table = pa.Table.from_pylist(values, schema=ASSET_SCHEMAS[filename])
        pq.write_table(table, path, compression="zstd", version="2.6")
        restored = pq.read_table(path)
        if restored.schema != ASSET_SCHEMAS[filename] or restored.to_pylist() != values:
            raise ValueError(f"{filename} failed its exact round-trip check")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{digest}  {path}")


def main() -> None:
    """Parse command-line arguments and export the assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
