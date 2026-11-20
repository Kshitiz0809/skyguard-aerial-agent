import json

from skyguard.agent.graph import HIGH_RISK_NODE, MissionRuntime, build_graph, new_mission
from skyguard.config import get_settings
from skyguard.simulator.scenarios import load_scenario


def _run(scenario, samples_dir, overrides=None):
    settings = get_settings()
    sim, spec = load_scenario(scenario, overrides)
    rt = MissionRuntime.create(sim, settings)
    graph = build_graph(rt)
    state, config = new_mission(samples_dir / spec["image"], scenario)
    graph.invoke(state, config)
    return rt, graph, config


def test_low_risk_only_mission_runs_to_completion(samples_dir):
    rt, graph, config = _run("solar_farm_hotspot", samples_dir)
    snap = graph.get_state(config)
    assert not snap.next  # no interrupt
    s = snap.values
    assert s["perception"]["anomaly"] == "damaged_solar_panel"
    assert {e["tool"] for e in s["executed"]} == {"log_incident", "capture_snapshot"}
    assert s["summary"]["final_mode"] == "AUTO"
    assert [t["node"] for t in s["trace"]][-1] == "report"


def test_high_risk_pauses_then_resumes_on_approval(samples_dir):
    rt, graph, config = _run("substation_intruder", samples_dir)
    snap = graph.get_state(config)
    assert HIGH_RISK_NODE in snap.next
    pending = snap.values["pending_high"]
    assert [a["tool"] for a in pending] == ["reroute_drone"]
    # Nothing high-risk has hit the aircraft yet
    assert all(c["command"] != "reroute" for c in rt.sim.command_log)

    graph.update_state(config, {"hitl_decisions": {pending[0]["id"]: {"approved": True, "operator": "test", "note": "ok"}}})
    graph.invoke(None, config)
    s = graph.get_state(config).values
    assert "reroute_drone" in s["summary"]["actions_executed"]
    assert rt.sim.state.mode in {"GUIDED", "LOITER"}
    assert any(c["mav_cmd"] == "MAV_CMD_DO_REPOSITION" for c in rt.sim.command_log)


def test_operator_edit_is_rechecked_by_guardrail(samples_dir):
    rt, graph, config = _run("substation_intruder", samples_dir)
    a = graph.get_state(config).values["pending_high"][0]
    edited = {**a["args"], "alt_m": 5.0}  # operator tries to go below HV floor
    graph.update_state(config, {"hitl_decisions": {a["id"]: {"approved": True, "operator": "t", "args": edited}}})
    graph.invoke(None, config)
    assert rt.sim.state.target_alt_m == 15.0 or rt.sim.state.alt_m >= 15.0


def test_rejecting_mandatory_rtl_logs_safety_override(samples_dir):
    rt, graph, config = _run("low_battery_intruder", samples_dir)
    pending = graph.get_state(config).values["pending_high"]
    assert pending[0]["tool"] == "trigger_emergency_rtl" and pending[0]["mandatory"]
    graph.update_state(config, {"hitl_decisions": {pending[0]["id"]: {"approved": False, "operator": "t", "note": "testing"}}})
    graph.invoke(None, config)
    s = graph.get_state(config).values
    assert s["summary"]["safety_overrides"] == 1
    audit = [json.loads(line) for line in (get_settings().runs_dir / "audit.jsonl").read_text().splitlines()]
    assert any(r["event"] == "SAFETY_OVERRIDE" for r in audit)


def test_high_wind_scenario_forces_rtl_and_approval_executes_it(samples_dir):
    rt, graph, config = _run("high_wind_hv_lines", samples_dir)
    pending = graph.get_state(config).values["pending_high"]
    assert [a["tool"] for a in pending] == ["trigger_emergency_rtl"]
    graph.update_state(config, {"hitl_decisions": {pending[0]["id"]: {"approved": True, "operator": "t"}}})
    graph.invoke(None, config)
    assert rt.sim.state.mode == "RTL"
