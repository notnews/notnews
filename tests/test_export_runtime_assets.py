"""Smoke tests for the model-asset export environment."""

from __future__ import annotations

import subprocess
import sys


def test_exporter_dependencies_and_cli_are_available() -> None:
    """The all-groups development environment can run the exporter."""
    subprocess.run(
        [sys.executable, "-c", "import joblib, sklearn"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "model_training/export_runtime_assets.py", "--help"],
        check=True,
    )
