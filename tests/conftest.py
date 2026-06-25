"""Shared pytest setup: make `lang_memory` importable and isolate each test's storage."""

import os
import sys

import pytest

# put the repo root (which contains the lang_memory package) on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lang_memory import config as lm_config


@pytest.fixture(autouse=True)
def isolate_storage(tmp_path, monkeypatch):
    """Every test reads/writes to its own temp dir — no shared state, no pollution."""
    monkeypatch.setattr(lm_config.settings, "DATA_DIR", str(tmp_path))
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: needs the Groq LLM (opt-in via RUN_LLM_TESTS=1)")
