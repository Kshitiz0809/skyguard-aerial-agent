"""Encode agent actions as real MAVLink v2 packets via pymavlink.

The packets are byte-exact MAVLink v2 frames (COMMAND_LONG / COMMAND_INT /
STATUSTEXT) that a PX4 or ArduPilot autopilot would accept. In this project
they are recorded rather than sent, but ``EncodedCommand.raw`` can be written
straight to a UDP socket pointed at SITL (e.g. udp:127.0.0.1:14550).

If pymavlink is not installed we still return the same command structure,
just without the encoded bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    from pymavlink.dialects.v20 import common as mavlink2

    _HAS_PYMAVLINK = True
except Exception:  # pragma: no cover - optional dependency
    mavlink2 = None
    _HAS_PYMAVLINK = False

# MAVLink enum values (kept here so the fallback path does not need pymavlink).
MAV_CMD_NAV_LOITER_UNLIM = 17
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_DO_CHANGE_ALTITUDE = 186
MAV_CMD_DO_REPOSITION = 192
MAV_CMD_IMAGE_START_CAPTURE = 2000
MAV_CMD_SET_CAMERA_ZOOM = 531
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6
MAV_DO_REPOSITION_FLAGS_CHANGE_MODE = 1
ZOOM_TYPE_RANGE = 2

GCS_SYSTEM_ID = 255
GCS_COMPONENT_ID = 190  # MAV_COMP_ID_MISSIONPLANNER
AUTOPILOT_COMPONENT_ID = 1

_CMD_NAMES = {
    MAV_CMD_NAV_LOITER_UNLIM: "MAV_CMD_NAV_LOITER_UNLIM",
    MAV_CMD_NAV_RETURN_TO_LAUNCH: "MAV_CMD_NAV_RETURN_TO_LAUNCH",
    MAV_CMD_DO_CHANGE_ALTITUDE: "MAV_CMD_DO_CHANGE_ALTITUDE",
    MAV_CMD_DO_REPOSITION: "MAV_CMD_DO_REPOSITION",
    MAV_CMD_IMAGE_START_CAPTURE: "MAV_CMD_IMAGE_START_CAPTURE",
    MAV_CMD_SET_CAMERA_ZOOM: "MAV_CMD_SET_CAMERA_ZOOM",
}


@dataclass
class EncodedCommand:
    message: str  # COMMAND_LONG | COMMAND_INT | STATUSTEXT
    command: str  # human-readable MAV_CMD name
    fields: dict[str, Any] = field(default_factory=dict)
    raw: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mavlink_message": self.message,
            "mav_cmd": self.command,
            "fields": self.fields,
            "mavlink_hex": self.raw.hex(" ") if self.raw else None,
            "mavlink_bytes": len(self.raw) if self.raw else 0,
        }


class _Encoder:
    def __init__(self) -> None:
        self.mav = mavlink2.MAVLink(None, srcSystem=GCS_SYSTEM_ID, srcComponent=GCS_COMPONENT_ID) if _HAS_PYMAVLINK else None

    def pack(self, msg) -> bytes:
        raw = msg.pack(self.mav)
        self.mav.seq = (self.mav.seq + 1) % 256
        return bytes(raw)


_encoder = _Encoder()


def has_pymavlink() -> bool:
    return _HAS_PYMAVLINK


def _command_long(target: int, command: int, params: list[float]) -> EncodedCommand:
    params = (params + [0.0] * 7)[:7]
    fields = {
        "target_system": target,
        "target_component": AUTOPILOT_COMPONENT_ID,
        "command": command,
        "confirmation": 0,
        **{f"param{i + 1}": p for i, p in enumerate(params)},
    }
    raw = None
    if _HAS_PYMAVLINK:
        msg = _encoder.mav.command_long_encode(target, AUTOPILOT_COMPONENT_ID, command, 0, *params)
        raw = _encoder.pack(msg)
    return EncodedCommand("COMMAND_LONG", _CMD_NAMES.get(command, str(command)), _jsonable(fields), raw)


def _command_int(target: int, command: int, frame: int, params: list[float], lat: float, lon: float, alt: float) -> EncodedCommand:
    params = (params + [0.0] * 4)[:4]
    x, y = int(round(lat * 1e7)), int(round(lon * 1e7))
    fields = {
        "target_system": target,
        "target_component": AUTOPILOT_COMPONENT_ID,
        "frame": frame,
        "command": command,
        "current": 0,
        "autocontinue": 0,
        **{f"param{i + 1}": p for i, p in enumerate(params)},
        "x": x,
        "y": y,
        "z": float(alt),
    }
    raw = None
    if _HAS_PYMAVLINK:
        msg = _encoder.mav.command_int_encode(
            target, AUTOPILOT_COMPONENT_ID, frame, command, 0, 0, *params, x, y, float(alt)
        )
        raw = _encoder.pack(msg)
    return EncodedCommand("COMMAND_INT", _CMD_NAMES.get(command, str(command)), _jsonable(fields), raw)


def _jsonable(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in fields.items()}


# ---------------------------------------------------------------- public API
def return_to_launch(target: int) -> EncodedCommand:
    return _command_long(target, MAV_CMD_NAV_RETURN_TO_LAUNCH, [])


def loiter(target: int) -> EncodedCommand:
    return _command_long(target, MAV_CMD_NAV_LOITER_UNLIM, [0, 0, 0, float("nan"), 0, 0, 0])


def reposition(target: int, lat: float, lon: float, alt_m: float) -> EncodedCommand:
    # param1 ground speed (-1 = default), param2 flags, param4 yaw (NaN = unchanged)
    return _command_int(
        target,
        MAV_CMD_DO_REPOSITION,
        MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
        [-1.0, float(MAV_DO_REPOSITION_FLAGS_CHANGE_MODE), 0.0, float("nan")],
        lat,
        lon,
        alt_m,
    )


def change_altitude(target: int, alt_m: float) -> EncodedCommand:
    return _command_long(target, MAV_CMD_DO_CHANGE_ALTITUDE, [float(alt_m), float(MAV_FRAME_GLOBAL_RELATIVE_ALT)])


def camera_zoom(target: int, zoom_level: float) -> EncodedCommand:
    # ZOOM_TYPE_RANGE expects 0..100 % of the zoom range; map 1x..10x onto it.
    pct = round((zoom_level - 1.0) / 9.0 * 100.0, 1)
    return _command_long(target, MAV_CMD_SET_CAMERA_ZOOM, [float(ZOOM_TYPE_RANGE), pct])


def capture_image(target: int) -> EncodedCommand:
    # param2 interval 0, param3 total images 1, param4 sequence 0
    return _command_long(target, MAV_CMD_IMAGE_START_CAPTURE, [0.0, 0.0, 1.0, 0.0])


def status_text(target: int, text: str, severity: int = 4) -> EncodedCommand:
    text = text[:50]
    fields = {"severity": severity, "text": text}
    raw = None
    if _HAS_PYMAVLINK:
        msg = _encoder.mav.statustext_encode(severity, text.encode("ascii", "replace"))
        raw = _encoder.pack(msg)
    return EncodedCommand("STATUSTEXT", "STATUSTEXT", fields, raw)
