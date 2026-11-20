"""Lightweight drone state simulator.

Tracks GPS position, altitude, battery, wind, flight mode and a camera frame
queue. Every command the agent issues is also encoded as a real MAVLink v2
packet (see ``mavlink.py``) so the output could be piped to a real autopilot
or SITL instance.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

from . import mavlink as mav

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class SectorInfo:
    name: str = "Substation-B"
    near_hv_lines: bool = True
    geofence_radius_m: float = 1500.0
    restricted_zones: list[str] = field(default_factory=list)


@dataclass
class DroneState:
    lat: float = 28.6139
    lon: float = 77.2090
    alt_m: float = 45.0
    heading_deg: float = 90.0
    ground_speed_ms: float = 6.0
    battery_pct: float = 64.0
    wind_kts: float = 8.0
    mode: str = "AUTO"  # AUTO | GUIDED | LOITER | RTL | LAND
    armed: bool = True
    camera_zoom: float = 1.0
    home_lat: float = 28.6129
    home_lon: float = 77.2080
    target_lat: float | None = None
    target_lon: float | None = None
    target_alt_m: float | None = None
    sector: SectorInfo = field(default_factory=SectorInfo)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("lat", "lon"):
            d[k] = round(d[k], 7)
        for k in ("alt_m", "battery_pct", "heading_deg"):
            d[k] = round(d[k], 1)
        d["distance_home_m"] = round(haversine_m(self.lat, self.lon, self.home_lat, self.home_lon), 1)
        d["sector_name"] = self.sector.name
        return d


class DroneSimulator:
    """~100-line kinematic + battery model with MAVLink command output."""

    # Battery burn (% per second) at hover / cruise, scaled up by wind.
    HOVER_BURN = 0.02
    CRUISE_BURN = 0.035

    def __init__(self, state: DroneState | None = None, system_id: int = 1):
        self.state = state or DroneState()
        self.system_id = system_id
        self.frame_queue: deque[str] = deque(maxlen=32)
        self.command_log: list[dict[str, Any]] = []
        self.sim_time_s = 0.0

    # ----------------------------------------------------------- frames
    def push_frame(self, image_path: str) -> None:
        self.frame_queue.append(image_path)

    def next_frame(self) -> str | None:
        return self.frame_queue.popleft() if self.frame_queue else None

    # ----------------------------------------------------------- physics
    def tick(self, dt: float = 1.0) -> None:
        s = self.state
        self.sim_time_s += dt
        moving = s.target_lat is not None and s.mode in {"GUIDED", "RTL", "AUTO"}
        burn = (self.CRUISE_BURN if moving else self.HOVER_BURN) * (1 + s.wind_kts / 40)
        s.battery_pct = max(0.0, s.battery_pct - burn * dt)

        if s.target_alt_m is not None:
            step = math.copysign(min(2.0 * dt, abs(s.target_alt_m - s.alt_m)), s.target_alt_m - s.alt_m)
            s.alt_m += step

        if moving:
            dist = haversine_m(s.lat, s.lon, s.target_lat, s.target_lon)
            travel = s.ground_speed_ms * dt
            if dist <= travel:
                s.lat, s.lon = s.target_lat, s.target_lon
                s.target_lat = s.target_lon = None
                if s.mode == "RTL":
                    s.mode, s.target_alt_m = "LAND", 0.0
                else:
                    s.mode = "LOITER"
            else:
                frac = travel / dist
                s.heading_deg = math.degrees(math.atan2(s.target_lon - s.lon, s.target_lat - s.lat)) % 360
                s.lat += (s.target_lat - s.lat) * frac
                s.lon += (s.target_lon - s.lon) * frac

        if s.mode == "LAND" and s.alt_m <= 0.1:
            s.alt_m, s.armed = 0.0, False

    def run(self, seconds: float, dt: float = 1.0) -> None:
        for _ in range(int(seconds / dt)):
            self.tick(dt)

    # ----------------------------------------------------------- commands
    def _record(self, name: str, packet: mav.EncodedCommand) -> dict[str, Any]:
        entry = {"t": round(self.sim_time_s, 1), "wall_time": time.time(), "command": name, **packet.to_dict()}
        self.command_log.append(entry)
        return entry

    def goto(self, lat: float, lon: float, alt_m: float) -> dict[str, Any]:
        s = self.state
        s.mode, s.target_lat, s.target_lon, s.target_alt_m = "GUIDED", lat, lon, alt_m
        return self._record("reroute", mav.reposition(self.system_id, lat, lon, alt_m))

    def change_altitude(self, alt_m: float) -> dict[str, Any]:
        self.state.target_alt_m = alt_m
        return self._record("change_altitude", mav.change_altitude(self.system_id, alt_m))

    def return_to_launch(self) -> dict[str, Any]:
        s = self.state
        s.mode, s.target_lat, s.target_lon = "RTL", s.home_lat, s.home_lon
        s.target_alt_m = max(s.alt_m, 30.0)  # climb to RTL altitude first
        return self._record("return_to_launch", mav.return_to_launch(self.system_id))

    def loiter(self) -> dict[str, Any]:
        s = self.state
        s.mode, s.target_lat, s.target_lon = "LOITER", None, None
        return self._record("loiter", mav.loiter(self.system_id))

    def set_zoom(self, level: float) -> dict[str, Any]:
        self.state.camera_zoom = max(1.0, min(level, 10.0))
        return self._record("set_camera_zoom", mav.camera_zoom(self.system_id, self.state.camera_zoom))

    def capture_image(self) -> dict[str, Any]:
        return self._record("capture_image", mav.capture_image(self.system_id))

    def status_text(self, text: str, severity: int = 4) -> dict[str, Any]:
        return self._record("statustext", mav.status_text(self.system_id, text, severity))

    def telemetry(self) -> dict[str, Any]:
        return self.state.to_dict()
