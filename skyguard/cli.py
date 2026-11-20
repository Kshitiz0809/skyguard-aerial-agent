"""Headless mission runner with terminal human-in-the-loop.

    python -m skyguard.cli --scenario substation_intruder
    python -m skyguard.cli --scenario low_battery_intruder --vlm mock --llm rules --auto approve
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent.graph import HIGH_RISK_NODE, MissionRuntime, build_graph, new_mission
from .config import get_settings
from .observability import setup_tracing
from .simulator.scenarios import SCENARIOS, load_scenario


def _print(title: str, obj) -> None:
    print(f"\n=== {title} ===")
    if isinstance(obj, list):
        for item in obj:
            print("  -", item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(f"  {k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)}")
    else:
        print(obj)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a SkyGuard mission")
    ap.add_argument("--scenario", default="substation_intruder", choices=list(SCENARIOS))
    ap.add_argument("--image", help="Override the scenario image path")
    ap.add_argument("--vlm", help="groq | qwen_local | hf_api | mock")
    ap.add_argument("--llm", help="groq | ollama | rules")
    ap.add_argument("--battery", type=float, help="Override battery %%")
    ap.add_argument("--wind", type=float, help="Override wind (knots)")
    ap.add_argument("--auto", choices=["approve", "reject"], help="Non-interactive HITL decision")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles + the § in SOP citations
    settings = get_settings()
    tracing = setup_tracing(settings)
    overrides = {k: v for k, v in {"battery_pct": args.battery, "wind_kts": args.wind}.items() if v is not None}
    sim, spec = load_scenario(args.scenario, overrides)
    image = args.image or str(settings.samples_dir / spec["image"])

    rt = MissionRuntime.create(sim, settings, vlm_backend=args.vlm, llm_provider=args.llm)
    graph = build_graph(rt)
    state, config = new_mission(image, args.scenario)

    print(f"SkyGuard | mission {state['mission_id']} | {spec['title']}")
    print(f"VLM={rt.vlm.name}:{rt.vlm.model}  LLM={rt.planner.provider}:{rt.planner.model}  RAG={rt.store.backend}  tracing={tracing}")

    graph.invoke(state, config)
    snap = graph.get_state(config)
    s = snap.values
    _print("Telemetry", {k: s["telemetry"][k] for k in ("battery_pct", "wind_kts", "alt_m", "mode", "sector_name")})
    _print("Perception", {k: s["perception"].get(k) for k in ("anomaly", "confidence", "scene_summary", "backend", "latency_ms", "fallback_error")})
    _print("Retrieved SOP", [f"{p['citation']}: {p['text'][:90]}" for p in s["policies"]])
    _print("Plan", {"provider": s["plan"]["provider"], "reasoning": s["plan"]["reasoning"], "actions": [(a["tool"], a["args"]) for a in s["plan"]["actions"]]})
    _print("Guardrail", {"warnings": s["guardrail"]["warnings"], "verdicts": [(d["action"]["tool"], d["verdict"], d["rules"], d["notes"]) for d in s["guardrail"]["decisions"]]})
    _print("Auto-executed (low-risk)", [(e["tool"], e["result"].get("mavlink", {}).get("mav_cmd")) for e in s.get("executed", [])])

    if HIGH_RISK_NODE in snap.next:
        print("\n>>> HUMAN APPROVAL REQUIRED (graph paused at interrupt_before=execute_high_risk)")
        decisions = {}
        for a in s["pending_high"]:
            flag = " [MANDATORY]" if a.get("mandatory") else ""
            print(f"  - {a['tool']}{flag} {json.dumps(a['args'])}  rules={a.get('guardrail_rules')}")
            if args.auto:
                ok = args.auto == "approve"
            else:
                ok = input("    approve? [y/N] ").strip().lower() == "y"
            decisions[a["id"]] = {"approved": ok, "operator": "cli-operator", "note": "approved via CLI" if ok else "rejected via CLI"}
        graph.update_state(config, {"hitl_decisions": decisions})
        graph.invoke(None, config)
        s = graph.get_state(config).values

    for e in s.get("executed", []):
        mv = e["result"].get("mavlink") or {}
        if mv.get("mavlink_hex"):
            print(f"  MAVLink {mv['mav_cmd']:<32} {mv['mavlink_hex'][:60]}...")
    _print("Mission summary", s["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
