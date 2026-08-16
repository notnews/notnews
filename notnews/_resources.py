"""Resolve immutable news-classifier assets."""

from __future__ import annotations

import os
from pathlib import Path

from ._schemas import ASSET_SCHEMAS

HF_REPO = "gojiberries/notnews"
HF_REVISION = "5b03656fdfa1ebc603f1d34356b1876cbd3ff765"
MODEL_DIR_ENV = "NOTNEWS_MODEL_DIR"
ASSET_FILES = frozenset(ASSET_SCHEMAS)


def resolve_asset(filename: str) -> str:
    """Return a local path for a pinned model asset.

    Args:
        filename: Asset path within the model repository.

    Returns:
        Local artifact path.
    """
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
