import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def offline_env(monkeypatch, tmp_path):
    """Force fully offline, deterministic backends for every test."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    monkeypatch.setenv("VLM_BACKEND", "mock")
    monkeypatch.setenv("RAG_BACKEND", "keyword")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("SKYGUARD_RUNS_DIR", str(tmp_path / "runs"))
    yield


@pytest.fixture
def samples_dir() -> Path:
    d = ROOT / "data" / "samples"
    if not (d / "substation_vehicle_fence.png").exists():
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_samples.py")], check=True)
    return d
