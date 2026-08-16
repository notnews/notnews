"""Arrow schemas for portable model assets."""

import pyarrow as pa

ASSET_SCHEMAS = {
    "metadata.parquet": pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("ngram_min", pa.int8(), nullable=False),
            pa.field("ngram_max", pa.int8(), nullable=False),
            pa.field("class_labels", pa.list_(pa.string()), nullable=False),
        ]
    ),
    "vocabulary.parquet": pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("feature", pa.string(), nullable=False),
            pa.field("feature_index", pa.int32(), nullable=False),
        ]
    ),
    "estimators.parquet": pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("fold", pa.int8(), nullable=False),
            pa.field("output_index", pa.int16(), nullable=False),
            pa.field("intercept", pa.float64(), nullable=False),
            pa.field("coefficients", pa.list_(pa.float64()), nullable=False),
        ]
    ),
    "calibrators.parquet": pa.schema(
        [
            pa.field("model", pa.string(), nullable=False),
            pa.field("fold", pa.int8(), nullable=False),
            pa.field("output_index", pa.int16(), nullable=False),
            pa.field("x_thresholds", pa.list_(pa.float64()), nullable=False),
            pa.field("y_thresholds", pa.list_(pa.float64()), nullable=False),
        ]
    ),
}
