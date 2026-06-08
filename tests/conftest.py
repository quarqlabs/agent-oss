# Shared fixtures for the Argus Agent test suite.
# All tests run without network access or API keys.

import os
import sys
import asyncio

import numpy as np
import pytest

# ==========================================
# PATH SETUP
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ==========================================
# FAKE EMBEDDING GENERATOR
# ==========================================
def fake_embedding(text: str) -> list[float]:
    """Deterministic 1536-d embedding derived from the text hash."""
    rng = np.random.default_rng(seed=hash(text) % (2**32))
    vec = rng.standard_normal(1536).astype("float32")
    vec /= np.linalg.norm(vec)
    return vec.tolist()


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, tmp_path):
    """Set minimal env vars so agent.py can be imported without real keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
    monkeypatch.setenv("AGENT_ID", "test_agent")
    monkeypatch.setenv("USER_ID", "test_user")
    monkeypatch.setenv("LOCAL_MEMORY_ROOT", str(tmp_path / "memory"))


@pytest.fixture
def memory_dir(tmp_path):
    """Provides a clean temporary directory for memory storage."""
    mem_dir = tmp_path / "memory" / "test_agent"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return mem_dir


@pytest.fixture
def event_loop():
    """Provide a fresh event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
