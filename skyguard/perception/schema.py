"""Structured output contract for the VLM perception node."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["low", "medium", "high", "critical"]

PERCEPTION_PROMPT = """You are the perception module of an autonomous security & inspection drone.
Inspect this aerial/nadir camera frame.

Flight context: sector={sector}, altitude={alt_m:.0f} m AGL, near_high_voltage_lines={near_hv}.

Report ONLY what is visible. Look for: unauthorized vehicles or people near/inside fences,
perimeter breaches, fire/smoke, vegetation encroaching on power lines, damaged insulators,
sagging conductors, oil leaks, damaged solar panels, obstacles/hazards to the aircraft.

Return a single JSON object, no prose, with exactly these keys:
{{
  "scene_summary": "<one sentence>",
  "anomaly": "<primary anomaly in snake_case, or 'none'>",
  "confidence": <0.0-1.0 for the primary anomaly>,
  "anomalies": [
    {{"type": "<snake_case>", "description": "<short>", "confidence": <0-1>,
      "severity": "low|medium|high|critical", "location_in_frame": "<e.g. top-left, center>"}}
  ],
  "objects": [{{"label": "<noun>", "count": <int>}}],
  "aircraft_hazards": ["<e.g. high_voltage_lines, tall_structure>"]
}}"""


def to_snake(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v).lower()).strip("_") or "unknown"


def clamp01(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


class Anomaly(BaseModel):
    type: str
    description: str = ""
    confidence: float = Field(default=0.5)
    severity: Severity = "medium"
    location_in_frame: str = "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return clamp01(v)

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str:
        v = str(v or "medium").lower()
        return v if v in {"low", "medium", "high", "critical"} else "medium"

    @field_validator("type", mode="before")
    @classmethod
    def _snake(cls, v: Any) -> str:
        return to_snake(v)


class PerceptionResult(BaseModel):
    scene_summary: str = ""
    anomaly: str = "none"
    confidence: float = 0.0
    anomalies: list[Anomaly] = Field(default_factory=list)
    objects: list[dict[str, Any]] = Field(default_factory=list)
    aircraft_hazards: list[str] = Field(default_factory=list)
    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0
    raw_text: str = ""

    def normalise(self) -> "PerceptionResult":
        if self.anomalies:
            top = max(self.anomalies, key=lambda a: a.confidence)
            if self.anomaly in {"", "none"}:
                self.anomaly, self.confidence = top.type, top.confidence
        if self.anomaly != "none" and not self.anomalies:
            self.anomalies = [Anomaly(type=self.anomaly, confidence=self.confidence)]
        self.anomaly = to_snake(self.anomaly) if self.anomaly else "none"
        self.confidence = clamp01(self.confidence, 0.0)
        return self


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, depth = text.find("{"), 0
    if start == -1:
        raise ValueError("No JSON object in VLM output")
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in VLM output")


def parse_perception(text: str, backend: str = "", model: str = "") -> PerceptionResult:
    data = _extract_json(text)
    result = PerceptionResult.model_validate({**data, "backend": backend, "model": model, "raw_text": text[:4000]})
    return result.normalise()
