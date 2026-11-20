"""Tracing setup (LangSmith / Arize Phoenix) + local JSONL audit trail."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import Settings

_PHOENIX_STARTED = False


def setup_tracing(settings: Settings) -> dict[str, Any]:
    """Enable whichever tracer is configured. Returns a status dict for the UI."""
    status = {"langsmith": False, "phoenix": False, "project": None}
    if settings.langsmith_enabled:
        # LangChain/LangGraph auto-trace when these env vars are present.
        os.environ.setdefault("LANGSMITH_PROJECT", "skyguard")
        status["langsmith"] = True
        status["project"] = os.environ["LANGSMITH_PROJECT"]
    if settings.phoenix_enabled:
        status["phoenix"] = _setup_phoenix()
    return status


def _setup_phoenix() -> bool:
    global _PHOENIX_STARTED
    if _PHOENIX_STARTED:
        return True
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        tracer_provider = register(project_name="skyguard", auto_instrument=False)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        _PHOENIX_STARTED = True
        return True
    except Exception:
        return False


class AuditLog:
    """Append-only JSONL log of every detection, decision, approval and MAVLink command."""

    def __init__(self, runs_dir: Path):
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = runs_dir / "audit.jsonl"

    def write(self, event: str, mission_id: str, **payload: Any) -> None:
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, "mission_id": mission_id, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def tail(self, n: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
