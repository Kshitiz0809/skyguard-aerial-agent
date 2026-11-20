"""Central configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- Reasoning LLM ---------------------------------------------------
    # "groq" | "ollama" | "rules"  (rules = deterministic offline planner)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))

    # --- Vision-language model ------------------------------------------
    # "groq" | "qwen_local" | "hf_api" | "mock" | "auto"
    vlm_backend: str = field(default_factory=lambda: _env("VLM_BACKEND", "auto"))
    groq_vlm_model: str = field(
        default_factory=lambda: _env("GROQ_VLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    )
    qwen_local_model: str = field(default_factory=lambda: _env("QWEN_LOCAL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))
    hf_vlm_model: str = field(default_factory=lambda: _env("HF_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))

    # --- RAG -------------------------------------------------------------
    sop_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "sop")
    chroma_dir: Path = field(default_factory=lambda: Path(_env("CHROMA_DIR", str(ROOT_DIR / ".chroma"))))
    rag_backend: str = field(default_factory=lambda: _env("RAG_BACKEND", "auto"))  # chroma | keyword | auto
    rag_top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "4")))

    # --- Data / logs -----------------------------------------------------
    samples_dir: Path = field(default_factory=lambda: ROOT_DIR / "data" / "samples")
    runs_dir: Path = field(default_factory=lambda: Path(_env("SKYGUARD_RUNS_DIR", str(ROOT_DIR / "runs"))))

    # --- Observability ---------------------------------------------------
    langsmith_enabled: bool = field(
        default_factory=lambda: _env_bool("LANGSMITH_TRACING") and bool(_env("LANGSMITH_API_KEY"))
    )
    phoenix_enabled: bool = field(default_factory=lambda: _env_bool("PHOENIX_ENABLED"))

    def resolved_llm_provider(self) -> str:
        if self.llm_provider and self.llm_provider != "auto":
            return self.llm_provider
        return "groq" if self.groq_api_key else "rules"

    def resolved_vlm_backend(self) -> str:
        if self.vlm_backend and self.vlm_backend != "auto":
            return self.vlm_backend
        return "groq" if self.groq_api_key else "mock"


def get_settings() -> Settings:
    return Settings()
