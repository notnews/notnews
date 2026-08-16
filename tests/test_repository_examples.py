"""Smoke tests for runnable repository applications and examples."""

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace


def test_llm_example_imports_current_api() -> None:
    """The LLM example imports without executing API calls."""
    runpy.run_path("examples/llm_classification_example.py")


def test_streamlit_app_imports_current_api(monkeypatch) -> None:
    """The Streamlit app starts with the public package API."""
    streamlit = SimpleNamespace(
        button=lambda _label: False,
        error=lambda _message: None,
        file_uploader=lambda _label, **_kwargs: None,
        title=lambda _title: None,
        write=lambda _value: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit)
    runpy.run_path("streamlit/streamlit_app.py")
