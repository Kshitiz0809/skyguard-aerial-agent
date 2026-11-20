"""Agent tools, risk classes and executors.

LOW-RISK  tools never change the aircraft trajectory -> executed automatically.
HIGH-RISK tools change trajectory / mission state  -> LangGraph interrupt, human approval.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ..simulator.drone import DroneSimulator

LOW_RISK = "low"
HIGH_RISK = "high"


# ------------------------------------------------------------------ arg schemas
class LogIncidentArgs(BaseModel):
    incident_type: str = Field(description="snake_case anomaly type, e.g. unauthorized_vehicle_near_fence")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Severity per SOP")
    description: str = Field(description="One-sentence description of what was observed")


class ZoomCameraArgs(BaseModel):
    level: float = Field(ge=1.0, le=10.0, description="Optical zoom level 1x-10x")


class CaptureSnapshotArgs(BaseModel):
    note: str = Field(default="", description="Why the snapshot is taken")


class AlertSecurityArgs(BaseModel):
    message: str = Field(description="Alert text for the duty security officer")
    priority: Literal["P1", "P2", "P3"] = Field(default="P2", description="P1 = immediate dispatch")


class RerouteArgs(BaseModel):
    lat: float = Field(description="Target latitude (WGS84 decimal degrees)")
    lon: float = Field(description="Target longitude (WGS84 decimal degrees)")
    alt_m: float = Field(description="Target altitude in metres AGL")
    reason: str = Field(default="", description="Why the reroute is needed")


class ChangeAltitudeArgs(BaseModel):
    alt_m: float = Field(description="New altitude in metres AGL")
    reason: str = Field(default="")


class EmergencyRTLArgs(BaseModel):
    reason: str = Field(description="Why Return-to-Launch is required")


# ------------------------------------------------------------------ executors
def _incident_log_path(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / "incidents.jsonl"


def _exec_log_incident(sim: DroneSimulator, a: LogIncidentArgs, ctx: dict) -> dict:
    record = {
        "incident_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mission_id": ctx.get("mission_id"),
        "sector": sim.state.sector.name,
        "lat": round(sim.state.lat, 6),
        "lon": round(sim.state.lon, 6),
        **a.model_dump(),
    }
    runs_dir = ctx.get("runs_dir")
    if runs_dir:
        with _incident_log_path(Path(runs_dir)).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    packet = sim.status_text(f"{a.severity}:{a.incident_type}", severity=2 if a.severity == "CRITICAL" else 4)
    return {"incident": record, "mavlink": packet}


def _exec_zoom(sim: DroneSimulator, a: ZoomCameraArgs, ctx: dict) -> dict:
    return {"camera_zoom": a.level, "mavlink": sim.set_zoom(a.level)}


def _exec_snapshot(sim: DroneSimulator, a: CaptureSnapshotArgs, ctx: dict) -> dict:
    return {"snapshot": ctx.get("image_path"), "note": a.note, "mavlink": sim.capture_image()}


def _exec_alert(sim: DroneSimulator, a: AlertSecurityArgs, ctx: dict) -> dict:
    # Simulated notification channel. Swap in Slack/Twilio/MQTT here.
    return {"delivered_to": "duty_security_officer (simulated)", "priority": a.priority, "message": a.message}


def _exec_reroute(sim: DroneSimulator, a: RerouteArgs, ctx: dict) -> dict:
    return {"target": {"lat": a.lat, "lon": a.lon, "alt_m": a.alt_m}, "mavlink": sim.goto(a.lat, a.lon, a.alt_m)}


def _exec_change_alt(sim: DroneSimulator, a: ChangeAltitudeArgs, ctx: dict) -> dict:
    return {"target_alt_m": a.alt_m, "mavlink": sim.change_altitude(a.alt_m)}


def _exec_rtl(sim: DroneSimulator, a: EmergencyRTLArgs, ctx: dict) -> dict:
    return {"mode": "RTL", "reason": a.reason, "mavlink": sim.return_to_launch()}


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class ToolSpec:
    name: str
    risk: str
    description: str
    args_model: type[BaseModel]
    executor: Callable[[DroneSimulator, Any, dict], dict]
    changes_trajectory: bool = False


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("log_incident", LOW_RISK, "Write an incident record to the mission audit log.", LogIncidentArgs, _exec_log_incident),
        ToolSpec("zoom_camera", LOW_RISK, "Set gimbal camera optical zoom (1x-10x).", ZoomCameraArgs, _exec_zoom),
        ToolSpec("capture_snapshot", LOW_RISK, "Capture a high-resolution still for evidence.", CaptureSnapshotArgs, _exec_snapshot),
        ToolSpec("alert_security_team", LOW_RISK, "Notify the duty security officer.", AlertSecurityArgs, _exec_alert),
        ToolSpec("reroute_drone", HIGH_RISK, "Fly to a new GPS waypoint (MAV_CMD_DO_REPOSITION). Requires human approval.", RerouteArgs, _exec_reroute, True),
        ToolSpec("change_altitude", HIGH_RISK, "Climb/descend to a new altitude. Requires human approval.", ChangeAltitudeArgs, _exec_change_alt, True),
        ToolSpec("trigger_emergency_rtl", HIGH_RISK, "Abort mission and Return-to-Launch. Requires human approval.", EmergencyRTLArgs, _exec_rtl, True),
    ]
}


def make_action(tool: str, args: dict[str, Any], reason: str = "", source: str = "planner") -> dict[str, Any]:
    spec = TOOLS.get(tool)
    return {
        "id": uuid.uuid4().hex[:8],
        "tool": tool,
        "args": args,
        "reason": reason,
        "risk": spec.risk if spec else "unknown",
        "source": source,
    }


def llm_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas (accepted by ChatGroq / ChatOllama .bind_tools)."""
    out = []
    for spec in TOOLS.values():
        params = spec.args_model.model_json_schema()
        params.pop("title", None)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": f"[{spec.risk.upper()}-RISK] {spec.description}",
                    "parameters": params,
                },
            }
        )
    return out


def execute_action(sim: DroneSimulator, action: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    spec = TOOLS[action["tool"]]
    args = spec.args_model.model_validate(action["args"])
    started = time.perf_counter()
    result = spec.executor(sim, args, ctx)
    return {
        "action_id": action["id"],
        "tool": spec.name,
        "risk": spec.risk,
        "args": args.model_dump(),
        "status": "executed",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "result": result,
    }
