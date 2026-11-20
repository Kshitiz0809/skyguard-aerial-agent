"""Offline tests for the multi-turn planner loop and tiled VLM merging (no API calls)."""

from langchain_core.messages import AIMessage
from PIL import Image

from skyguard.agent.planner import Planner
from skyguard.config import get_settings
from skyguard.perception.schema import Anomaly, PerceptionResult
from skyguard.perception.vlm import VLMBackend, make_tiles, merge_results
from skyguard.simulator.scenarios import load_scenario


# ------------------------------------------------------------------ planner loop
class ScriptedLLM:
    """Mimics a model that emits one tool call per turn (like gpt-oss on Groq)."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def bind(self, **_):
        return self

    def invoke(self, messages):
        self.calls += 1
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _tc(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def _planner(script):
    p = Planner(get_settings(), "rules")
    p.provider, p.llm = "groq", object()
    p.llm_with_tools = ScriptedLLM(script)
    return p


def _inputs():
    sim, _ = load_scenario("substation_intruder")
    perception = {"anomaly": "unauthorized_vehicle_near_fence", "confidence": 0.9, "anomalies": []}
    return perception, sim.telemetry(), []


def test_tool_loop_collects_one_call_per_turn_and_dedupes():
    log = {"incident_type": "v", "severity": "HIGH", "description": "d"}
    p = _planner([
        _tc("log_incident", log, 1),
        _tc("zoom_camera", {"level": 4}, 2),
        _tc("zoom_camera", {"level": 4}, 3),  # duplicate -> ignored
        _tc("alert_security_team", {"message": "m", "priority": "P2"}, 4),
        AIMessage(content="Done per SOP-SUB-B-v1.3 §4.1."),
    ])
    out = p.plan(*_inputs())
    assert [a["tool"] for a in out["actions"]] == ["log_incident", "zoom_camera", "alert_security_team"]
    assert out["llm_turns"] == 5 and "§4.1" in out["reasoning"] and out["provider"] == "groq"


def test_tool_loop_retries_malformed_tool_json():
    bad = RuntimeError("Error code: 400 - tool_use_failed: Failed to parse tool call arguments as JSON")
    p = _planner([bad, _tc("zoom_camera", {"level": 2}, 1), AIMessage(content="ok")])
    out = p.plan(*_inputs())
    assert [a["tool"] for a in out["actions"]] == ["zoom_camera"]
    assert "retried" in " ".join(out.get("notes", []))


def test_tool_loop_keeps_partial_plan_when_later_step_fails():
    boom = RuntimeError("connection reset")
    p = _planner([_tc("zoom_camera", {"level": 2}, 1), boom])
    out = p.plan(*_inputs())
    assert out["provider"] == "groq" and [a["tool"] for a in out["actions"]] == ["zoom_camera"]


def test_tool_loop_first_step_failure_falls_back_to_rules():
    p = _planner([RuntimeError("401 invalid api key")])
    out = p.plan(*_inputs())
    assert out["provider"] == "rules (fallback)" and "401" in out["fallback_error"]
    assert any(a["tool"] == "log_incident" for a in out["actions"])


# ------------------------------------------------------------------ tiling
def test_make_tiles_overlap_and_upscale():
    tiles = make_tiles(Image.new("RGB", (1024, 768)), grid=2, out_side=1024)
    assert [n for n, _ in tiles] == ["top-left", "top-right", "bottom-left", "bottom-right"]
    assert all(max(t.size) == 1024 for _, t in tiles)


def test_merge_promotes_tile_only_critical_anomaly():
    full = PerceptionResult(scene_summary="substation", anomaly="none").normalise()
    tile = PerceptionResult(anomalies=[Anomaly(type="unauthorized_person", confidence=0.9, severity="critical")]).normalise()
    weak = PerceptionResult(anomalies=[Anomaly(type="bird", confidence=0.2)]).normalise()
    merged = merge_results(full, [("top-left", tile), ("top-right", weak)])
    assert merged.anomaly == "unauthorized_person" and merged.anomalies[0].location_in_frame == "top-left"
    assert all(a.type != "bird" for a in merged.anomalies)  # below min confidence
    assert "Tiled pass added" in merged.scene_summary


class CountingVLM(VLMBackend):
    name, model = "fake", "fake"

    def __init__(self):
        super().__init__(tiling=True)
        self.prompts = []

    def _infer(self, img, prompt):
        self.prompts.append(prompt)
        if "ZOOMED-IN TILE covering the bottom-left" in prompt:
            return PerceptionResult(anomalies=[Anomaly(type="person_inside_fence", confidence=0.8, severity="critical")]).normalise()
        return PerceptionResult(scene_summary="yard", anomaly="none").normalise()


def test_tiled_analyze_runs_full_plus_four_tiles(samples_dir):
    vlm = CountingVLM()
    sim, _ = load_scenario("low_battery_intruder")
    r = vlm.analyze(samples_dir / "substation_person_fence.png", sim.telemetry())
    assert len(vlm.prompts) == 5 and r.tiles_analyzed == 4
    assert r.anomaly == "person_inside_fence"
