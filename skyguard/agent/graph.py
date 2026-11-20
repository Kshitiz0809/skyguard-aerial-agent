"""SkyGuard LangGraph mission loop.

    ingest -> perceive -> retrieve_policy -> plan -> guardrail -> execute_low_risk
                                                                     |
                                       (pending high-risk?) ---------+---------> report -> END
                                                |
                                   [interrupt_before] execute_high_risk -> report -> END

The graph is compiled with ``interrupt_before=["execute_high_risk"]`` and a
checkpointer. When high-risk actions are pending, execution halts. The
operator's decisions are written into state with ``graph.update_state`` and
the run resumes with ``graph.invoke(None, config)``.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from ..observability import AuditLog
from ..perception.vlm import MockVLM, VLMBackend, build_vlm
from ..rag.policy_store import PolicyStore
from ..safety.guardrails import GuardrailLimits, apply_guardrails
from ..simulator.drone import DroneSimulator
from ..tools.actions import execute_action
from .planner import Planner

HIGH_RISK_NODE = "execute_high_risk"


class MissionState(TypedDict, total=False):
    mission_id: str
    scenario: str
    image_path: str
    telemetry: dict[str, Any]
    perception: dict[str, Any]
    policy_queries: list[str]
    policies: list[dict[str, Any]]
    plan: dict[str, Any]
    guardrail: dict[str, Any]
    pending_high: list[dict[str, Any]]
    hitl_decisions: dict[str, dict[str, Any]]
    executed: Annotated[list[dict[str, Any]], operator.add]
    rejected: Annotated[list[dict[str, Any]], operator.add]
    safety_overrides: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    summary: dict[str, Any]


@dataclass
class MissionRuntime:
    """Non-serialisable dependencies shared by graph nodes."""

    sim: DroneSimulator
    vlm: VLMBackend
    store: PolicyStore
    planner: Planner
    settings: Settings = field(default_factory=get_settings)
    limits: GuardrailLimits = field(default_factory=GuardrailLimits)
    audit: AuditLog | None = None

    def __post_init__(self) -> None:
        if self.audit is None:
            self.audit = AuditLog(self.settings.runs_dir)

    @classmethod
    def create(
        cls,
        sim: DroneSimulator,
        settings: Settings | None = None,
        vlm_backend: str | None = None,
        llm_provider: str | None = None,
        rag_backend: str | None = None,
        store: PolicyStore | None = None,
    ) -> "MissionRuntime":
        settings = settings or get_settings()
        store = store or PolicyStore(settings.sop_dir, settings.chroma_dir, rag_backend or settings.rag_backend)
        return cls(
            sim=sim,
            vlm=build_vlm(settings, vlm_backend),
            store=store,
            planner=Planner(settings, llm_provider),
            settings=settings,
        )


def _traced(name: str, fn: Callable[[MissionState], dict]) -> Callable[[MissionState], dict]:
    @wraps(fn)
    def wrapper(state: MissionState) -> dict:
        started = time.perf_counter()
        update = fn(state) or {}
        event = {"node": name, "ms": round((time.perf_counter() - started) * 1000, 1), "detail": update.pop("_detail", "")}
        update["trace"] = [event]
        return update

    return wrapper


def build_graph(rt: MissionRuntime, checkpointer=None):
    audit = rt.audit

    # --------------------------------------------------------------- nodes
    def ingest(state: MissionState) -> dict:
        rt.sim.push_frame(state["image_path"])
        frame = rt.sim.next_frame()
        telemetry = rt.sim.telemetry()
        audit.write("mission_start", state["mission_id"], scenario=state.get("scenario"), image=frame, telemetry=telemetry)
        return {"image_path": frame, "telemetry": telemetry, "_detail": f"frame={Path(frame).name} battery={telemetry['battery_pct']:.0f}%"}

    def perceive(state: MissionState) -> dict:
        try:
            result = rt.vlm.analyze(state["image_path"], state["telemetry"])
            perception = result.model_dump()
        except Exception as exc:
            # A failed VLM call must not stall the safety loop: fall back and say so.
            result = MockVLM().analyze(state["image_path"], state["telemetry"])
            perception = {**result.model_dump(), "fallback_error": f"{type(exc).__name__}: {exc}"[:400]}
        audit.write("perception", state["mission_id"], anomaly=perception["anomaly"], confidence=perception["confidence"], backend=perception["backend"])
        return {"perception": perception, "_detail": f"{perception['anomaly']} ({perception['confidence']:.2f}) via {perception['backend']}"}

    def retrieve_policy(state: MissionState) -> dict:
        p, t = state["perception"], state["telemetry"]
        queries = []
        if p["anomaly"] != "none":
            descs = " ".join(a.get("description", "") for a in p.get("anomalies", [])[:2])
            queries.append(f"{p['anomaly'].replace('_', ' ')} {descs} response procedure")
        else:
            queries.append("routine patrol standard altitude logging")
        if t["battery_pct"] < 30:
            queries.append("battery low return to launch RTL threshold")
        if t["wind_kts"] >= 18:
            queries.append("wind knots limit stop patrol near HV lines altitude")
        if t["sector"].get("near_hv_lines"):
            queries.append("minimum altitude high-voltage lines conductors separation")
        queries.append("high-risk actions require human approval reroute altitude RTL")
        hits = rt.store.multi_query(queries, k=rt.settings.rag_top_k)[: rt.settings.rag_top_k + 3]
        policies = [h.to_dict() for h in hits]
        return {"policy_queries": queries, "policies": policies, "_detail": f"{len(policies)} clauses via {rt.store.backend}"}

    def plan(state: MissionState) -> dict:
        out = rt.planner.plan(state["perception"], state["telemetry"], state["policies"])
        audit.write("plan", state["mission_id"], provider=out["provider"], actions=[(a["tool"], a["args"]) for a in out["actions"]])
        return {"plan": out, "_detail": f"{len(out['actions'])} actions via {out['provider']}"}

    def guardrail(state: MissionState) -> dict:
        report = apply_guardrails(state["plan"]["actions"], state["telemetry"], rt.limits)
        audit.write(
            "guardrail",
            state["mission_id"],
            warnings=report["warnings"],
            verdicts=[(d["action"]["tool"], d["verdict"], d["rules"]) for d in report["decisions"]],
        )
        detail = f"{len(report['approved_low'])} auto, {len(report['pending_high'])} need approval, {len(report['blocked'])} blocked"
        return {"guardrail": report, "pending_high": report["pending_high"], "_detail": detail}

    def execute_low_risk(state: MissionState) -> dict:
        ctx = {"mission_id": state["mission_id"], "image_path": state["image_path"], "runs_dir": rt.settings.runs_dir}
        results = []
        for action in state["guardrail"]["approved_low"]:
            res = execute_action(rt.sim, action, ctx)
            res["approved_by"] = "auto (low-risk)"
            results.append(res)
            audit.write("executed", state["mission_id"], tool=res["tool"], args=res["args"], approved_by=res["approved_by"],
                        mavlink=(res["result"].get("mavlink") or {}).get("mav_cmd"))
        rt.sim.tick(1.0)
        return {"executed": results, "telemetry": rt.sim.telemetry(), "_detail": f"{len(results)} low-risk executed"}

    def execute_high_risk(state: MissionState) -> dict:
        decisions = state.get("hitl_decisions") or {}
        ctx = {"mission_id": state["mission_id"], "image_path": state["image_path"], "runs_dir": rt.settings.runs_dir}
        executed, rejected, overrides = [], [], []
        for action in state.get("pending_high", []):
            d = decisions.get(action["id"], {"approved": False, "note": "no operator decision", "operator": "none"})
            if d.get("approved"):
                # Operator may have edited the arguments in the UI; re-check them against the guardrail.
                final = {**action, "args": d.get("args") or action["args"]}
                recheck = apply_guardrails([final], rt.sim.telemetry(), rt.limits)
                passed = next((a for a in recheck["pending_high"] if a["id"] == action["id"]), None)
                if passed is None:
                    notes = [n for x in recheck["decisions"] for n in x["notes"]]
                    rejected.append({**action, "decision": d, "status": "blocked_on_recheck", "notes": notes})
                    audit.write("blocked_on_recheck", state["mission_id"], tool=action["tool"], notes=notes)
                    continue
                res = execute_action(rt.sim, passed, ctx)
                res["approved_by"] = d.get("operator", "operator")
                res["operator_note"] = d.get("note", "")
                executed.append(res)
                audit.write("executed", state["mission_id"], tool=res["tool"], args=res["args"], approved_by=res["approved_by"],
                            note=res["operator_note"], mavlink=(res["result"].get("mavlink") or {}).get("mav_cmd"))
            else:
                rejected.append({**action, "decision": d, "status": "rejected"})
                audit.write("rejected", state["mission_id"], tool=action["tool"], args=action["args"], decision=d)
                if action.get("mandatory"):
                    ov = {"action": action["tool"], "rules": action.get("guardrail_rules"), "operator": d.get("operator"), "note": d.get("note")}
                    overrides.append(ov)
                    audit.write("SAFETY_OVERRIDE", state["mission_id"], **ov)
        rt.sim.run(5.0)
        return {
            "executed": executed,
            "rejected": rejected,
            "safety_overrides": overrides,
            "telemetry": rt.sim.telemetry(),
            "_detail": f"{len(executed)} approved+executed, {len(rejected)} rejected",
        }

    def report(state: MissionState) -> dict:
        p = state["perception"]
        summary = {
            "mission_id": state["mission_id"],
            "anomaly": p["anomaly"],
            "confidence": p["confidence"],
            "actions_executed": [e["tool"] for e in state.get("executed", [])],
            "actions_rejected": [r["tool"] for r in state.get("rejected", [])],
            "actions_blocked": [b["action"]["tool"] for b in state["guardrail"]["blocked"]],
            "safety_overrides": len(state.get("safety_overrides", [])),
            "mavlink_commands": len(rt.sim.command_log),
            "final_mode": rt.sim.state.mode,
            "final_battery_pct": round(rt.sim.state.battery_pct, 1),
            "total_ms": round(sum(t["ms"] for t in state.get("trace", [])), 1),
        }
        audit.write("mission_end", state["mission_id"], **{k: v for k, v in summary.items() if k != "mission_id"})
        return {"summary": summary, "_detail": f"mode={summary['final_mode']}"}

    def route_after_low(state: MissionState) -> str:
        return HIGH_RISK_NODE if state.get("pending_high") else "report"

    # --------------------------------------------------------------- wiring
    g = StateGraph(MissionState)
    for name, fn in [
        ("ingest", ingest),
        ("perceive", perceive),
        ("retrieve_policy", retrieve_policy),
        ("plan", plan),
        ("guardrail", guardrail),
        ("execute_low_risk", execute_low_risk),
        (HIGH_RISK_NODE, execute_high_risk),
        ("report", report),
    ]:
        g.add_node(name, _traced(name, fn))
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "perceive")
    g.add_edge("perceive", "retrieve_policy")
    g.add_edge("retrieve_policy", "plan")
    g.add_edge("plan", "guardrail")
    g.add_edge("guardrail", "execute_low_risk")
    g.add_conditional_edges("execute_low_risk", route_after_low, {HIGH_RISK_NODE: HIGH_RISK_NODE, "report": "report"})
    g.add_edge(HIGH_RISK_NODE, "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=[HIGH_RISK_NODE])


def new_mission(image_path: str | Path, scenario: str = "custom") -> tuple[MissionState, dict]:
    mission_id = f"MSN-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
    state: MissionState = {"mission_id": mission_id, "scenario": scenario, "image_path": str(image_path)}
    config = {"configurable": {"thread_id": mission_id}, "run_name": f"skyguard:{scenario}", "tags": ["skyguard", scenario]}
    return state, config
