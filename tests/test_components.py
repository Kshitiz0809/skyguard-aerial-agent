import pytest

from skyguard.config import get_settings
from skyguard.perception.schema import parse_perception
from skyguard.rag.policy_store import PolicyStore
from skyguard.safety.guardrails import apply_guardrails
from skyguard.simulator import mavlink
from skyguard.simulator.drone import DroneSimulator, DroneState, haversine_m
from skyguard.simulator.scenarios import load_scenario
from skyguard.tools.actions import TOOLS, execute_action, llm_tool_schemas, make_action


# ------------------------------------------------------------------ simulator
def test_simulator_reroute_and_rtl():
    sim = DroneSimulator(DroneState(lat=28.6139, lon=77.2090, battery_pct=50))
    sim.goto(28.6145, 77.2095, 40)
    sim.run(30)
    assert sim.state.mode == "LOITER"
    assert haversine_m(sim.state.lat, sim.state.lon, 28.6145, 77.2095) < 1
    assert sim.state.battery_pct < 50
    sim.return_to_launch()
    sim.run(400)
    assert sim.state.mode == "LAND" and not sim.state.armed


def test_mavlink_packets_are_v2():
    if not mavlink.has_pymavlink():
        pytest.skip("pymavlink not installed")
    pkt = mavlink.return_to_launch(1)
    assert pkt.raw[0] == 0xFD  # MAVLink v2 start-of-frame
    assert pkt.command == "MAV_CMD_NAV_RETURN_TO_LAUNCH"
    rep = mavlink.reposition(1, 28.6, 77.2, 30)
    assert rep.message == "COMMAND_INT" and rep.fields["x"] == 286000000
    # Round-trip decode
    from pymavlink.dialects.v20 import common as m2

    decoded = m2.MAVLink(None).parse_char(rep.raw)
    assert decoded.get_type() == "COMMAND_INT" and decoded.command == mavlink.MAV_CMD_DO_REPOSITION


# ------------------------------------------------------------------ perception
def test_parse_perception_handles_fenced_and_noisy_json():
    text = 'Sure! ```json\n{"anomaly": "Unauthorized Vehicle Near Fence", "confidence": 1.7, "anomalies": []}\n```'
    p = parse_perception(text)
    assert p.anomaly == "unauthorized_vehicle_near_fence"
    assert p.confidence == 1.0
    assert p.anomalies and p.anomalies[0].type == "unauthorized_vehicle_near_fence"


def test_parse_perception_promotes_top_anomaly():
    p = parse_perception('{"anomalies": [{"type": "a", "confidence": 0.3}, {"type": "smoke", "confidence": 0.8, "severity": "CRITICAL"}]}')
    assert p.anomaly == "smoke" and p.confidence == 0.8 and p.anomalies[1].severity == "critical"


# ------------------------------------------------------------------ RAG
def test_policy_store_keyword_retrieval():
    s = get_settings()
    store = PolicyStore(s.sop_dir, None, "keyword")
    hits = store.query("battery below 20 return to launch", k=3)
    assert any(h.clause == "6.2" for h in hits)
    hits = store.query("descend near high voltage lines minimum altitude", k=3)
    assert any(h.clause in {"4.4", "D.1"} for h in hits)
    assert hits[0].citation.startswith(("SOP-", "FOM-"))


# ------------------------------------------------------------------ tools
def test_tool_schemas_and_risk_split():
    names = {t["function"]["name"] for t in llm_tool_schemas()}
    assert {"log_incident", "zoom_camera", "reroute_drone", "trigger_emergency_rtl"} <= names
    assert TOOLS["zoom_camera"].risk == "low" and TOOLS["reroute_drone"].risk == "high"


def test_execute_low_risk_tool(tmp_path):
    sim = DroneSimulator()
    res = execute_action(sim, make_action("zoom_camera", {"level": 4}), {"runs_dir": tmp_path})
    assert res["status"] == "executed" and sim.state.camera_zoom == 4
    assert res["result"]["mavlink"]["mav_cmd"] == "MAV_CMD_SET_CAMERA_ZOOM"


# ------------------------------------------------------------------ guardrails
def _telemetry(**kw):
    sim, _ = load_scenario("substation_intruder", kw)
    return sim.telemetry()


def test_guardrail_battery_critical_forces_rtl_and_blocks_reroute():
    t = _telemetry(battery_pct=15)
    plan = [make_action("reroute_drone", {"lat": t["lat"], "lon": t["lon"], "alt_m": 40}), make_action("log_incident", {"incident_type": "x", "severity": "HIGH", "description": "d"})]
    r = apply_guardrails(plan, t)
    assert r["rtl_forced"]
    assert [b["action"]["tool"] for b in r["blocked"]] == ["reroute_drone"]
    assert [a["tool"] for a in r["pending_high"]] == ["trigger_emergency_rtl"]
    assert r["pending_high"][0]["mandatory"] and r["pending_high"][0]["source"] == "guardrail"
    assert [a["tool"] for a in r["approved_low"]] == ["log_incident"]


def test_guardrail_clamps_altitude_near_hv_lines():
    t = _telemetry(battery_pct=80, wind_kts=5)
    r = apply_guardrails([make_action("change_altitude", {"alt_m": 8})], t)
    assert r["decisions"][0]["verdict"] == "modify"
    assert r["pending_high"][0]["args"]["alt_m"] == 15
    assert any("4.4" in rule for rule in r["decisions"][0]["rules"])


def test_guardrail_wind_caution_raises_floor_and_ceiling_clamp():
    t = _telemetry(battery_pct=80, wind_kts=20)
    r = apply_guardrails([make_action("change_altitude", {"alt_m": 20}), make_action("change_altitude", {"alt_m": 300})], t)
    alts = [a["args"]["alt_m"] for a in r["pending_high"]]
    assert alts == [30, 120]


def test_guardrail_geofence_and_low_battery_detour():
    t = _telemetry(battery_pct=80)
    far = make_action("reroute_drone", {"lat": t["lat"] + 0.05, "lon": t["lon"], "alt_m": 40})
    assert apply_guardrails([far], t)["blocked"][0]["rules"][0].endswith("§7.2")
    t = _telemetry(battery_pct=25)
    near = make_action("reroute_drone", {"lat": t["lat"] + 0.0002, "lon": t["lon"], "alt_m": 40})
    assert apply_guardrails([near], t)["blocked"][0]["rules"][0].endswith("§6.1")


def test_guardrail_blocks_unknown_tool_and_bad_args():
    t = _telemetry()
    r = apply_guardrails([make_action("self_destruct", {}), make_action("zoom_camera", {"level": 50})], t)
    assert {b["rules"][0] for b in r["blocked"]} == {"UNKNOWN_TOOL", "INVALID_ARGS"}


def test_guardrail_high_wind_injects_rtl():
    t = _telemetry(battery_pct=60, wind_kts=30)
    r = apply_guardrails([make_action("zoom_camera", {"level": 2})], t)
    assert r["rtl_forced"] and r["pending_high"][0]["tool"] == "trigger_emergency_rtl"
