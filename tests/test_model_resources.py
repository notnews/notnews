"""Contracts for published notnews assets."""

from pathlib import Path
from unittest.mock import patch

import pytest

from notnews._resources import ASSET_FILES, HF_REPO, HF_REVISION, resolve_asset


def test_local_override_avoids_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit model directory takes precedence over the Hub."""
    artifact = tmp_path / "metadata.parquet"
    artifact.write_bytes(b"model")
    monkeypatch.setenv("NOTNEWS_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_asset("metadata.parquet") == str(artifact)
    download.assert_not_called()


def test_missing_local_artifact_uses_exact_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback download uses the declared repository and revision."""
    monkeypatch.setenv("NOTNEWS_MODEL_DIR", str(tmp_path))

    with patch(
        "huggingface_hub.hf_hub_download", return_value="/cache/metadata.parquet"
    ) as download:
        assert resolve_asset("metadata.parquet") == "/cache/metadata.parquet"
    download.assert_called_once_with(HF_REPO, "metadata.parquet", revision=HF_REVISION)


def test_revision_is_an_immutable_commit() -> None:
    """Hub revision pins use the full hexadecimal commit identifier."""
    assert len(HF_REVISION) == 40
    assert set(HF_REVISION) <= set("0123456789abcdef")


@pytest.mark.live
def test_pinned_revision_contains_every_artifact() -> None:
    """The published snapshot contains every portable model table."""
    from huggingface_hub import list_repo_files

    published = set(list_repo_files(HF_REPO, revision=HF_REVISION))
    assert ASSET_FILES <= published
