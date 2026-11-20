"""Deterministic safety guardrail.

The LLM *proposes*; this module *disposes*. Every rule here mirrors a clause
of the site SOP / flight manual (cited by ID) and is plain code, so the
safety envelope does not depend on the LLM following instructions.

Verdicts per proposed action:
  allow   – passes unchanged
  modify  – arguments clamped into the safe envelope (e.g. altitude floor)
  block   – removed from the plan
Mandatory actions (e.g. battery-critical RTL) are *injected* when missing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..simulator.drone import haversine_m
from ..tools.actions import TOOLS, make_action

SOP = "SOP-SUB-B-v1.3"
FOM = "FOM-REG-SUMMARY-v0.9"


@dataclass(frozen=True)
class GuardrailLimits:
    battery_rtl_pct: float = 20.0  # SOP 6.2 / FOM C.2
    battery_no_detour_pct: float = 30.0  # SOP 6.1
    wind_abort_kts: float = 25.0  # SOP 6.3 / FOM C.4
    wind_caution_kts: float = 18.0  # SOP 6.3
    hv_min_alt_m: float = 15.0  # SOP 4.4 / FOM D.1
    hv_wind_min_alt_m: float = 30.0  # SOP 6.3
    max_alt_m: float = 120.0  # SOP 7.1 / FOM A.1


def _decision(action: dict, verdict: str, rules: list[str], notes: list[str], final_args: dict | None) -> dict[str, Any]:
    return {
        "action": action,
        "verdict": verdict,
        "rules": rules,
        "notes": notes,
        "final_args": final_args,
    }


def apply_guardrails(
    actions: list[dict[str, Any]],
    telemetry: dict[str, Any],
    limits: GuardrailLimits = GuardrailLimits(),
) -> dict[str, Any]:
    battery = float(telemetry.get("battery_pct", 100))
    wind = float(telemetry.get("wind_kts", 0))
    sector = telemetry.get("sector", {}) or {}
    near_hv = bool(sector.get("near_hv_lines", False))
    geofence = float(sector.get("geofence_radius_m", 1e9))
    home = (telemetry.get("home_lat"), telemetry.get("home_lon"))

    decisions: list[dict[str, Any]] = []
    warnings: list[str] = []
    mandatory: list[dict[str, Any]] = []

    battery_critical = battery < limits.battery_rtl_pct
    wind_abort = wind > limits.wind_abort_kts
    if battery_critical:
        warnings.append(f"Battery {battery:.0f}% < {limits.battery_rtl_pct:.0f}%: RTL mandatory, trajectory changes cancelled ({SOP} §6.2).")
    elif battery < limits.battery_no_detour_pct:
        warnings.append(f"Battery {battery:.0f}% < {limits.battery_no_detour_pct:.0f}%: no new detours ({SOP} §6.1).")
    if wind_abort:
        warnings.append(f"Wind {wind:.0f} kt > {limits.wind_abort_kts:.0f} kt: patrol must stop and RTL ({SOP} §6.3, {FOM} §C.4).")
    elif wind >= limits.wind_caution_kts and near_hv:
        warnings.append(f"Wind {wind:.0f} kt near HV lines: altitude floor {limits.hv_wind_min_alt_m:.0f} m ({SOP} §6.3).")

    rtl_forced = battery_critical or wind_abort
    rtl_proposed = any(a.get("tool") == "trigger_emergency_rtl" for a in actions)
    seen_rtl = False

    for action in actions:
        tool = action.get("tool")
        spec = TOOLS.get(tool)
        if spec is None:
            decisions.append(_decision(action, "block", ["UNKNOWN_TOOL"], [f"Tool '{tool}' is not in the registry."], None))
            continue
        try:
            args = spec.args_model.model_validate(action.get("args", {})).model_dump()
        except ValidationError as exc:
            decisions.append(_decision(action, "block", ["INVALID_ARGS"], [str(exc.errors()[0].get("msg", exc))], None))
            continue

        rules: list[str] = []
        notes: list[str] = []
        verdict = "allow"

        if tool == "trigger_emergency_rtl":
            if seen_rtl:
                decisions.append(_decision(action, "block", ["DUPLICATE"], ["RTL already in plan."], None))
                continue
            seen_rtl = True
            if rtl_forced:
                rules.append(f"{SOP} §6.2" if battery_critical else f"{SOP} §6.3")
                notes.append("Mandatory: operator rejection is logged as a SAFETY OVERRIDE.")
                action = {**action, "mandatory": True}

        elif spec.changes_trajectory:
            # RTL supersedes any other trajectory change.
            if rtl_forced or rtl_proposed:
                why = "battery critical" if battery_critical else "wind above limit" if wind_abort else "RTL in plan"
                decisions.append(
                    _decision(action, "block", [f"{SOP} §6.2" if battery_critical else f"{SOP} §6.3"], [f"Cancelled: {why}; RTL takes precedence."], None)
                )
                continue
            if tool == "reroute_drone" and battery < limits.battery_no_detour_pct:
                decisions.append(_decision(action, "block", [f"{SOP} §6.1"], [f"Battery {battery:.0f}%: no new detours."], None))
                continue
            if tool == "reroute_drone" and None not in home:
                dist = haversine_m(args["lat"], args["lon"], home[0], home[1])
                if dist > geofence:
                    decisions.append(
                        _decision(action, "block", [f"{SOP} §7.2", f"{FOM} §A.2"], [f"Target {dist:.0f} m from home exceeds geofence {geofence:.0f} m."], None)
                    )
                    continue
            # Altitude envelope
            floor = 0.0
            if near_hv:
                floor = limits.hv_min_alt_m
                if wind >= limits.wind_caution_kts:
                    floor = limits.hv_wind_min_alt_m
            if args["alt_m"] < floor:
                rules.append(f"{SOP} §6.3" if floor == limits.hv_wind_min_alt_m else f"{SOP} §4.4")
                notes.append(f"Altitude {args['alt_m']:.0f} m below floor {floor:.0f} m near HV lines -> clamped.")
                args["alt_m"] = floor
                verdict = "modify"
            if args["alt_m"] > limits.max_alt_m:
                rules.append(f"{SOP} §7.1")
                notes.append(f"Altitude {args['alt_m']:.0f} m above {limits.max_alt_m:.0f} m ceiling -> clamped.")
                args["alt_m"] = limits.max_alt_m
                verdict = "modify"

        if not rules:
            rules.append(f"{SOP} §3.1" if spec.risk == "low" else f"{SOP} §3.2")
        decisions.append(_decision(action, verdict, rules, notes, args))

    if rtl_forced and not seen_rtl:
        reason = (
            f"Battery {battery:.0f}% below {limits.battery_rtl_pct:.0f}% RTL threshold"
            if battery_critical
            else f"Wind {wind:.0f} kt exceeds {limits.wind_abort_kts:.0f} kt limit"
        )
        injected = make_action("trigger_emergency_rtl", {"reason": reason}, reason=reason, source="guardrail")
        injected["mandatory"] = True
        mandatory.append(injected)
        decisions.append(
            _decision(injected, "inject", [f"{SOP} §6.2" if battery_critical else f"{SOP} §6.3"], ["Injected by guardrail (mandatory)."], deepcopy(injected["args"]))
        )

    approved_low, pending_high, blocked = [], [], []
    for d in decisions:
        if d["verdict"] == "block":
            blocked.append(d)
            continue
        final = {**d["action"], "args": d["final_args"], "guardrail_rules": d["rules"], "guardrail_notes": d["notes"]}
        (approved_low if TOOLS[final["tool"]].risk == "low" else pending_high).append(final)

    return {
        "decisions": decisions,
        "warnings": warnings,
        "approved_low": approved_low,
        "pending_high": pending_high,
        "blocked": blocked,
        "rtl_forced": rtl_forced,
    }
