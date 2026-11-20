"""Reasoning planner: turns perception + telemetry + retrieved SOP into tool calls.

* ``groq``   – Llama-3.3-70B via Groq (native parallel tool calling, sub-second).
* ``ollama`` – any local tool-calling model (qwen2.5:7b, llama3.1:8b, ...).
* ``rules``  – deterministic SOP playbook. Offline default and LLM-failure fallback.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import Settings
from ..tools.actions import llm_tool_schemas, make_action

SYSTEM_PROMPT = """You are SkyGuard, the mission-reasoning agent of an autonomous inspection & security drone.

You receive: (1) structured VLM perception of the current camera frame, (2) live telemetry,
(3) retrieved clauses from the Site SOP and flight-operations manual.

Decide what to do NOW by calling tools. Rules:
- Ground every decision in the retrieved clauses; mention clause IDs (e.g. "SOP-SUB-B-v1.3 §4.1") in your text.
- Call ALL tools you want executed in this single response (parallel tool calls are fine).
- LOW-RISK tools (log_incident, zoom_camera, capture_snapshot, alert_security_team) run automatically.
- HIGH-RISK tools (reroute_drone, change_altitude, trigger_emergency_rtl) pause for human approval;
  only propose them when the SOP calls for it. Reroute targets must be real lat/lon near the current position.
- If no anomaly is present, call no tools and say the patrol continues.
- A deterministic safety guardrail will validate your plan afterwards; do not try to bypass limits.
Respond with one short paragraph of reasoning plus the tool calls."""


def _context_message(perception: dict, telemetry: dict, policies: list[dict]) -> str:
    perception_view = {k: perception.get(k) for k in ("scene_summary", "anomaly", "confidence", "anomalies", "aircraft_hazards")}
    telemetry_view = {
        k: telemetry.get(k)
        for k in ("lat", "lon", "alt_m", "battery_pct", "wind_kts", "mode", "camera_zoom", "distance_home_m", "home_lat", "home_lon")
    }
    telemetry_view["sector"] = telemetry.get("sector")
    clauses = "\n".join(f"- [{p['citation']}] ({p['section']}) {p['text']}" for p in policies)
    return (
        f"PERCEPTION:\n{json.dumps(perception_view, indent=1)}\n\n"
        f"TELEMETRY:\n{json.dumps(telemetry_view, indent=1)}\n\n"
        f"RETRIEVED POLICY CLAUSES:\n{clauses or '(none)'}"
    )


# ----------------------------------------------------------------- rules
def rules_plan(perception: dict, telemetry: dict, policies: list[dict]) -> dict[str, Any]:
    """Deterministic playbook transcribed from SOP §4-§6."""
    anomaly = perception.get("anomaly", "none") or "none"
    conf = float(perception.get("confidence", 0))
    top = max(perception.get("anomalies") or [{}], key=lambda a: a.get("confidence", 0))
    desc = top.get("description") or anomaly.replace("_", " ")
    lat, lon, alt = telemetry["lat"], telemetry["lon"], telemetry["alt_m"]
    battery, wind = telemetry.get("battery_pct", 100), telemetry.get("wind_kts", 0)
    actions: list[dict] = []
    why: list[str] = []
    A = lambda tool, args, r: actions.append(make_action(tool, args, reason=r, source="rules"))  # noqa: E731

    if battery < 20:
        A("trigger_emergency_rtl", {"reason": f"Battery {battery:.0f}% below RTL threshold"}, "SOP §6.2")
        why.append("Battery critical: RTL is mandatory (SOP-SUB-B-v1.3 §6.2).")
    elif wind > 25:
        A("trigger_emergency_rtl", {"reason": f"Wind {wind:.0f} kt above 25 kt limit"}, "SOP §6.3")
        why.append("Wind above airframe limit: stop patrol and RTL (SOP §6.3, FOM §C.4).")

    if anomaly == "none" or conf < 0.35:
        why.append("No actionable anomaly; patrol continues.")
        return {"reasoning": " ".join(why), "actions": actions}

    a = anomaly
    if "vehicle" in a:
        A("log_incident", {"incident_type": a, "severity": "HIGH", "description": desc}, "SOP §4.1")
        A("zoom_camera", {"level": 4.0}, "SOP §4.1 plate capture")
        A("capture_snapshot", {"note": "Registration plate evidence"}, "SOP §4.1")
        A("alert_security_team", {"message": f"Vehicle at fence, {telemetry['sector']['name']}: {desc}", "priority": "P2"}, "SOP §4.1")
        if battery >= 30:
            A("reroute_drone", {"lat": lat + 0.0003, "lon": lon + 0.0003, "alt_m": 30.0, "reason": "30 m standoff for closer look"}, "SOP §4.3")
        why.append("Vehicle near fence: log HIGH, zoom >=4x, snapshot, alert DSO (SOP §4.1); propose standoff reroute (SOP §4.3).")
    elif "person" in a or "intrud" in a or "breach" in a:
        A("log_incident", {"incident_type": a, "severity": "CRITICAL", "description": desc}, "SOP §4.2")
        A("alert_security_team", {"message": f"PERSON inside fence at {telemetry['sector']['name']}: {desc}", "priority": "P1"}, "SOP §4.2")
        A("zoom_camera", {"level": 3.0}, "SOP §4.2 keep visual contact")
        A("capture_snapshot", {"note": "Intruder evidence"}, "SOP §4.2")
        why.append("Person inside fence: CRITICAL incident, immediate P1 alert, keep visual contact without descending (SOP §4.2).")
    elif "fire" in a or "smoke" in a:
        A("log_incident", {"incident_type": a, "severity": "CRITICAL", "description": desc}, "SOP §5.3")
        A("alert_security_team", {"message": f"Smoke/fire at {telemetry['sector']['name']}", "priority": "P1"}, "SOP §5.3")
        if battery >= 30:
            A("reroute_drone", {"lat": lat + 0.0009, "lon": lon - 0.0009, "alt_m": max(alt, 50.0), "reason": "Upwind 100 m standoff"}, "SOP §5.3")
        why.append("Fire/smoke: CRITICAL, alert, propose upwind standoff (SOP §5.3).")
    elif any(k in a for k in ("vegetation", "conductor", "insulator", "oil", "sag")):
        sev = "HIGH" if "oil" in a else "MEDIUM"
        A("log_incident", {"incident_type": a, "severity": sev, "description": desc}, "SOP §5.1")
        A("zoom_camera", {"level": 5.0}, "SOP §5.1")
        A("capture_snapshot", {"note": "Asset defect evidence"}, "SOP §5.1")
        why.append(f"Asset defect: log {sev}, zoom and snapshot (SOP §5.1).")
    else:
        A("log_incident", {"incident_type": a, "severity": "LOW" if "solar" in a else "MEDIUM", "description": desc}, "SOP §5.2")
        A("capture_snapshot", {"note": "Inspection evidence"}, "SOP §5.2")
        why.append("Inspection finding: log and snapshot; no flight-path change needed (SOP §5.2).")
    return {"reasoning": " ".join(why), "actions": actions}


# ------------------------------------------------------------------- LLM
class Planner:
    def __init__(self, settings: Settings, provider: str | None = None):
        self.provider = provider or settings.resolved_llm_provider()
        self.model = ""
        self.llm = None
        if self.provider == "groq":
            from langchain_groq import ChatGroq

            if not settings.groq_api_key:
                raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEY is not set in .env")
            self.model = settings.groq_model
            self.llm = ChatGroq(model=self.model, api_key=settings.groq_api_key, temperature=0.1, max_retries=2)
        elif self.provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pip install langchain-ollama to use LLM_PROVIDER=ollama") from exc
            self.model = settings.ollama_model
            self.llm = ChatOllama(model=self.model, base_url=settings.ollama_base_url, temperature=0.1)
        elif self.provider == "rules":
            self.model = "sop-playbook"
        else:
            raise ValueError(f"Unknown LLM provider '{self.provider}'")
        if self.llm is not None:
            self.llm_with_tools = self.llm.bind_tools(llm_tool_schemas())

    def plan(self, perception: dict, telemetry: dict, policies: list[dict]) -> dict[str, Any]:
        started = time.perf_counter()
        if self.llm is None:
            out = rules_plan(perception, telemetry, policies)
            return {**out, "provider": "rules", "model": self.model, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            msg = self.llm_with_tools.invoke(
                [SystemMessage(SYSTEM_PROMPT), HumanMessage(_context_message(perception, telemetry, policies))]
            )
            actions = [make_action(tc["name"], tc.get("args", {}), reason="llm", source=self.provider) for tc in msg.tool_calls]
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            usage = getattr(msg, "usage_metadata", None) or {}
            return {
                "reasoning": content.strip() or "(model returned tool calls only)",
                "actions": actions,
                "provider": self.provider,
                "model": self.model,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "tokens": {k: usage.get(k) for k in ("input_tokens", "output_tokens")} if usage else None,
            }
        except Exception as exc:  # network / rate limit / malformed tool call -> safe fallback
            out = rules_plan(perception, telemetry, policies)
            return {
                **out,
                "provider": "rules (fallback)",
                "model": "sop-playbook",
                "fallback_error": f"{type(exc).__name__}: {exc}"[:500],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
