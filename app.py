"""SkyGuard operator console (Streamlit).

    streamlit run app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from skyguard.agent.graph import HIGH_RISK_NODE, MissionRuntime, build_graph, new_mission
from skyguard.agent.planner import Planner
from skyguard.config import get_settings
from skyguard.observability import AuditLog, setup_tracing
from skyguard.perception.vlm import build_vlm
from skyguard.rag.policy_store import PolicyStore
from skyguard.simulator.scenarios import SCENARIOS, load_scenario

st.set_page_config(page_title="SkyGuard Operator Console", page_icon="🛰️", layout="wide")

st.markdown(
    """
<style>
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;margin-right:6px}
.pill-ok{background:#1f7a4d22;color:#1f9d5c;border:1px solid #1f9d5c55}
.pill-warn{background:#b8860b22;color:#c99700;border:1px solid #c9970055}
.pill-bad{background:#c0392b22;color:#e0533f;border:1px solid #e0533f55}
.pill-info{background:#2b6cb022;color:#4a90d9;border:1px solid #4a90d955}
.small{font-size:0.85rem;opacity:0.8}
</style>
""",
    unsafe_allow_html=True,
)

settings = get_settings()
tracing = setup_tracing(settings)


# ------------------------------------------------------------------ cached deps
@st.cache_resource(show_spinner="Indexing SOP into the vector store…")
def get_store(backend: str) -> PolicyStore:
    return PolicyStore(settings.sop_dir, settings.chroma_dir, backend)


@st.cache_resource(show_spinner="Loading vision-language model…")
def get_vlm(backend: str):
    return build_vlm(settings, backend)


@st.cache_resource
def get_planner(provider: str) -> Planner:
    return Planner(settings, provider)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛰️ SkyGuard")
    st.caption("Aerial VLM + autonomous mission agent")

    st.subheader("Models")
    has_groq = bool(settings.groq_api_key)
    vlm_options = (["groq"] if has_groq else []) + ["mock", "qwen_local"] + (["hf_api"] if settings.hf_token else [])
    llm_options = (["groq"] if has_groq else []) + ["rules", "ollama"]
    default_vlm = settings.resolved_vlm_backend()
    default_llm = settings.resolved_llm_provider()
    vlm_choice = st.selectbox("Vision model (VLM)", vlm_options, index=vlm_options.index(default_vlm) if default_vlm in vlm_options else 0)
    llm_choice = st.selectbox("Reasoning LLM", llm_options, index=llm_options.index(default_llm) if default_llm in llm_options else 0)
    tiling = st.toggle(
        "Thorough small-object scan",
        value=settings.vlm_tiling_for(vlm_choice),
        disabled=vlm_choice == "mock",
        help="Full frame + 4 zoomed tiles (5 VLM calls), merged SAHI-style. Catches ~25 px people the "
        "full frame misses. About 1 min on Groq's free tier (8k tokens/min).",
    )
    if not has_groq:
        st.info("Add `GROQ_API_KEY` to `.env` to enable the Groq VLM and the Llama-3.3-70B planner. Offline mode uses the mock VLM and the rules planner.")

    st.subheader("Scenario")
    scenario = st.selectbox("Preset", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["title"])
    spec = SCENARIOS[scenario]
    st.caption(spec["description"])

    with st.expander("Telemetry overrides", expanded=False):
        battery = st.slider("Battery %", 5, 100, int(spec["state"]["battery_pct"]))
        wind = st.slider("Wind (knots)", 0, 40, int(spec["state"]["wind_kts"]))
        alt = st.slider("Altitude (m AGL)", 10, 120, int(spec["state"]["alt_m"]))
        near_hv = st.toggle("Near HV lines", value=spec["sector"]["near_hv_lines"])

    st.subheader("Camera frame")
    sample_files = sorted(p for p in settings.samples_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    names = [str(p.relative_to(settings.samples_dir)) for p in sample_files]
    default_img = spec["image"] if spec["image"] in names else (names[0] if names else None)
    picked = st.selectbox("Sample frame", names, index=names.index(default_img) if default_img else 0) if names else None
    upload = st.file_uploader("…or upload an aerial image", type=["png", "jpg", "jpeg"])

    launch = st.button("🚀 Launch mission", type="primary", width="stretch")

    st.divider()
    st.markdown(
        pill("LangSmith on" if tracing["langsmith"] else "LangSmith off", "ok" if tracing["langsmith"] else "info")
        + pill("Phoenix on" if tracing["phoenix"] else "Phoenix off", "ok" if tracing["phoenix"] else "info"),
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ launch
if launch:
    if upload is not None:
        up_dir = settings.runs_dir / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        image_path = up_dir / f"{int(time.time())}_{Path(upload.name).name}"
        image_path.write_bytes(upload.getvalue())
    elif picked:
        image_path = settings.samples_dir / picked
    else:
        st.error("No image available. Run `python scripts/generate_samples.py` first.")
        st.stop()

    try:
        sim, _ = load_scenario(scenario, {"battery_pct": float(battery), "wind_kts": float(wind), "alt_m": float(alt), "near_hv_lines": near_hv})
        vlm = get_vlm(vlm_choice)
        vlm.tiling = tiling and vlm_choice != "mock"
        rt = MissionRuntime(
            sim=sim,
            vlm=vlm,
            store=get_store(settings.rag_backend),
            planner=get_planner(llm_choice),
            settings=settings,
        )
    except Exception as exc:
        st.error(f"Could not initialise backends: {exc}")
        st.stop()

    graph = build_graph(rt)
    state, config = new_mission(image_path, scenario)
    with st.status(f"Running mission {state['mission_id']}…", expanded=False) as status:
        graph.invoke(state, config)
        status.update(label=f"Mission {state['mission_id']}: graph checkpoint reached", state="complete")
    st.session_state.update(rt=rt, graph=graph, config=config, mission_id=state["mission_id"])


# ------------------------------------------------------------------ main view
if "graph" not in st.session_state:
    st.title("SkyGuard Operator Console")
    st.markdown(
        """
**Pipeline:** camera frame + telemetry → **VLM perception** (structured JSON) → **RAG** over the site SOP & flight manual
→ **LLM planner** (tool calls) → **deterministic safety guardrail** → low-risk tools run automatically,
high-risk tools **pause the LangGraph run** for your approval → MAVLink commands + audit trail.

Pick a scenario in the sidebar and press **Launch mission**.
"""
    )
    st.stop()

graph, config, rt = st.session_state.graph, st.session_state.config, st.session_state.rt
snap = graph.get_state(config)
s = snap.values
paused = HIGH_RISK_NODE in (snap.next or ())
perception, telemetry, plan, guard = s.get("perception", {}), s.get("telemetry", {}), s.get("plan", {}), s.get("guardrail", {})

status_pill = pill("AWAITING OPERATOR", "warn") if paused else pill("MISSION COMPLETE", "ok")
sev = max((a.get("severity", "low") for a in perception.get("anomalies", [])), key=["low", "medium", "high", "critical"].index, default="low")
sev_kind = {"low": "info", "medium": "warn", "high": "bad", "critical": "bad"}[sev]
conf_text = "conf {:.2f}".format(perception.get("confidence", 0))
st.markdown(
    f"### Mission `{s['mission_id']}`  {status_pill}{pill(perception.get('anomaly', 'none'), sev_kind)}{pill(conf_text, 'info')}",
    unsafe_allow_html=True,
)

# ---- HITL panel (top of page while paused)
if paused:
    with st.container(border=True):
        st.subheader("🧑‍✈️ Human approval required")
        st.caption("LangGraph is paused at `interrupt_before=['execute_high_risk']`. Nothing below has been sent to the aircraft.")
        operator = st.text_input("Operator ID", value=st.session_state.get("operator", "RPIC-01"))
        decisions = {}
        for a in s.get("pending_high", []):
            with st.container(border=True):
                head = f"**`{a['tool']}`**"
                if a.get("mandatory"):
                    head += "  " + pill("MANDATORY", "bad")
                if a.get("source") == "guardrail":
                    head += pill("injected by guardrail", "warn")
                st.markdown(head, unsafe_allow_html=True)
                st.caption("Rules: " + ", ".join(a.get("guardrail_rules", [])) + ("  ·  " + " ".join(a["guardrail_notes"]) if a.get("guardrail_notes") else ""))
                args = dict(a["args"])
                if a["tool"] == "reroute_drone":
                    c1, c2, c3 = st.columns(3)
                    args["lat"] = c1.number_input("lat", value=float(args["lat"]), format="%.6f", key=f"lat_{a['id']}")
                    args["lon"] = c2.number_input("lon", value=float(args["lon"]), format="%.6f", key=f"lon_{a['id']}")
                    args["alt_m"] = c3.number_input("alt (m)", value=float(args["alt_m"]), step=1.0, key=f"alt_{a['id']}")
                    st.caption(f"Reason: {args.get('reason', '')}")
                elif a["tool"] == "change_altitude":
                    args["alt_m"] = st.number_input("alt (m)", value=float(args["alt_m"]), step=1.0, key=f"alt_{a['id']}")
                else:
                    st.code(json.dumps(args, indent=1), language="json")
                c1, c2 = st.columns([1, 3])
                choice = c1.radio("Decision", ["Approve", "Reject"], key=f"dec_{a['id']}", horizontal=True)
                note = c2.text_input("Note", key=f"note_{a['id']}", placeholder="optional justification")
                if a.get("mandatory") and choice == "Reject":
                    st.warning("Rejecting a mandatory action is logged as a **SAFETY OVERRIDE** for review (SOP §8.2).")
                decisions[a["id"]] = {"approved": choice == "Approve", "operator": operator, "note": note, "args": args}
        if st.button("✅ Submit decisions & resume mission", type="primary"):
            st.session_state.operator = operator
            graph.update_state(config, {"hitl_decisions": decisions})
            with st.spinner("Resuming graph…"):
                graph.invoke(None, config)
            st.rerun()

# ---- frame + telemetry
left, right = st.columns([3, 2])
with left:
    st.image(s["image_path"], caption=Path(s["image_path"]).name, width="stretch")
with right:
    t = rt.sim.telemetry()
    m1, m2, m3 = st.columns(3)
    m1.metric("Battery", f"{t['battery_pct']:.1f}%", delta=f"{t['battery_pct'] - s['telemetry']['battery_pct']:.1f}" if not paused else None)
    m2.metric("Wind", f"{t['wind_kts']:.0f} kt")
    m3.metric("Altitude", f"{t['alt_m']:.0f} m")
    m4, m5, m6 = st.columns(3)
    m4.metric("Mode", t["mode"])
    m5.metric("Zoom", f"{t['camera_zoom']:.0f}x")
    m6.metric("From home", f"{t['distance_home_m']:.0f} m")
    pts = [{"lat": t["lat"], "lon": t["lon"], "kind": "drone", "color": "#e0533f", "size": 14},
           {"lat": t["home_lat"], "lon": t["home_lon"], "kind": "home", "color": "#1f9d5c", "size": 12}]
    if t.get("target_lat"):
        pts.append({"lat": t["target_lat"], "lon": t["target_lon"], "kind": "target", "color": "#4a90d9", "size": 12})
    st.map(pd.DataFrame(pts), latitude="lat", longitude="lon", color="color", size="size", zoom=15)
    st.caption(f"Sector **{t['sector']['name']}** · HV lines nearby: {t['sector']['near_hv_lines']} · geofence {t['sector']['geofence_radius_m']:.0f} m")

# ---- pipeline detail
tab_p, tab_r, tab_plan, tab_g, tab_x, tab_a = st.tabs(["👁️ Perception", "📚 Policy (RAG)", "🧠 Plan", "🛡️ Guardrail", "📡 Execution & MAVLink", "🧾 Audit trail"])

with tab_p:
    st.markdown(f"**{perception.get('scene_summary', '')}**")
    tiles = perception.get("tiles_analyzed") or 0
    st.caption(
        f"Backend `{perception.get('backend')}` · model `{perception.get('model')}` · {perception.get('latency_ms')} ms"
        + (f" · tiled scan: full frame + {tiles} tiles" if tiles else "")
    )
    if perception.get("fallback_error"):
        st.warning(f"VLM call failed, fell back to mock: {perception['fallback_error']}")
    if perception.get("anomalies"):
        st.dataframe(pd.DataFrame(perception["anomalies"]), width="stretch", hide_index=True)
    with st.expander("Structured JSON output"):
        st.json({k: v for k, v in perception.items() if k != "raw_text"})

with tab_r:
    st.caption(f"Vector store: `{rt.store.backend}` · queries: " + " | ".join(s.get("policy_queries", [])))
    for p in s.get("policies", []):
        st.markdown(f"**{p['citation']}** · _{p['section']}_ · score {p['score']}")
        st.markdown(f"> {p['text']}")

with tab_plan:
    st.caption(f"Planner `{plan.get('provider')}` · model `{plan.get('model')}` · {plan.get('latency_ms')} ms" + (f" · tokens {plan['tokens']}" if plan.get("tokens") else ""))
    if plan.get("fallback_error"):
        st.warning(f"LLM call failed, used rules fallback: {plan['fallback_error']}")
    st.markdown(plan.get("reasoning", ""))
    st.dataframe(
        pd.DataFrame([{"tool": a["tool"], "risk": a["risk"], "args": json.dumps(a["args"])} for a in plan.get("actions", [])]),
        width="stretch",
        hide_index=True,
    )

with tab_g:
    for w in guard.get("warnings", []):
        st.warning(w)
    rows = [
        {"tool": d["action"]["tool"], "verdict": d["verdict"].upper(), "rules": ", ".join(d["rules"]), "notes": " ".join(d["notes"]), "final_args": json.dumps(d["final_args"])}
        for d in guard.get("decisions", [])
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_x:
    ex = s.get("executed", [])
    if ex:
        st.markdown("**Executed actions**")
        st.dataframe(
            pd.DataFrame([{"tool": e["tool"], "risk": e["risk"], "approved_by": e.get("approved_by"), "note": e.get("operator_note", ""), "args": json.dumps(e["args"])} for e in ex]),
            width="stretch",
            hide_index=True,
        )
    if s.get("rejected"):
        st.markdown("**Rejected / blocked on recheck**")
        st.dataframe(pd.DataFrame([{"tool": r["tool"], "status": r["status"], "operator": r["decision"].get("operator"), "note": r["decision"].get("note")} for r in s["rejected"]]), hide_index=True)
    if s.get("safety_overrides"):
        st.error(f"{len(s['safety_overrides'])} SAFETY OVERRIDE(S) logged for review.")
    st.markdown("**MAVLink v2 packets** (byte-exact, ready for SITL / autopilot)")
    cmds = rt.sim.command_log
    if cmds:
        st.dataframe(
            pd.DataFrame([{"t": c["t"], "command": c["command"], "message": c["mavlink_message"], "mav_cmd": c["mav_cmd"], "bytes": c["mavlink_bytes"], "hex": c["mavlink_hex"]} for c in cmds]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No MAVLink commands issued yet.")
    st.markdown("**Graph trace**")
    st.dataframe(pd.DataFrame(s.get("trace", [])), width="stretch", hide_index=True)
    if s.get("summary"):
        st.json(s["summary"])

with tab_a:
    rows = [r for r in AuditLog(settings.runs_dir).tail(300) if r.get("mission_id") == s["mission_id"]]
    st.caption(f"`runs/audit.jsonl`: {len(rows)} events for this mission")
    for r in rows:
        kind = "🔴" if r["event"] == "SAFETY_OVERRIDE" else "•"
        st.markdown(f"{kind} `{r['ts']}` **{r['event']}** " + json.dumps({k: v for k, v in r.items() if k not in {'ts', 'event', 'mission_id'}}, default=str)[:400])
