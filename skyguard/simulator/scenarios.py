"""Preset mission scenarios: telemetry + sector + which sample frame to inspect."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .drone import DroneSimulator, DroneState, SectorInfo

SCENARIOS: dict[str, dict[str, Any]] = {
    "substation_intruder": {
        "title": "Substation-B: vehicle at the perimeter fence",
        "description": "Routine patrol. A vehicle is parked inside the fence line next to the transformer yard.",
        "image": "substation_vehicle_fence.png",
        "state": dict(lat=28.61392, lon=77.20905, alt_m=45.0, battery_pct=64.0, wind_kts=9.0),
        "sector": dict(name="Substation-B", near_hv_lines=True, geofence_radius_m=1500.0),
    },
    "low_battery_intruder": {
        "title": "Substation-B: intruder detected, battery 18%",
        "description": "Same intrusion, but battery is below the 20% RTL threshold. The guardrail must force RTL.",
        "image": "substation_person_fence.png",
        "state": dict(lat=28.61410, lon=77.20950, alt_m=40.0, battery_pct=18.0, wind_kts=11.0),
        "sector": dict(name="Substation-B", near_hv_lines=True, geofence_radius_m=1500.0),
    },
    "high_wind_hv_lines": {
        "title": "Line corridor: 27 kt wind near HV lines",
        "description": "Sagging conductor on the 220 kV corridor. Wind is gusting above the operating limit.",
        "image": "powerline_vegetation.png",
        "state": dict(lat=28.61500, lon=77.21100, alt_m=22.0, battery_pct=24.0, wind_kts=27.0),
        "sector": dict(name="HV-Corridor-7", near_hv_lines=True, geofence_radius_m=2500.0),
    },
    "solar_farm_hotspot": {
        "title": "Solar farm: damaged panel row",
        "description": "Low-risk inspection finding. The agent should log and zoom without asking a human.",
        "image": "solar_panel_damage.png",
        "state": dict(lat=28.62010, lon=77.21500, alt_m=35.0, battery_pct=71.0, wind_kts=6.0),
        "sector": dict(name="Solar-Block-C", near_hv_lines=False, geofence_radius_m=2000.0),
    },
}


def load_scenario(name: str, overrides: dict[str, Any] | None = None) -> tuple[DroneSimulator, dict[str, Any]]:
    """Return a fresh simulator configured for ``name`` (plus optional telemetry overrides)."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Options: {list(SCENARIOS)}")
    spec = deepcopy(SCENARIOS[name])
    state_kwargs = {**spec["state"], **(overrides or {})}
    sector_kwargs = spec["sector"]
    if overrides and "sector_name" in overrides:
        sector_kwargs["name"] = overrides["sector_name"]
    if overrides and "near_hv_lines" in overrides:
        sector_kwargs["near_hv_lines"] = bool(overrides["near_hv_lines"])
    state_kwargs.pop("sector_name", None)
    state_kwargs.pop("near_hv_lines", None)
    sim = DroneSimulator(DroneState(sector=SectorInfo(**sector_kwargs), **state_kwargs))
    return sim, spec
