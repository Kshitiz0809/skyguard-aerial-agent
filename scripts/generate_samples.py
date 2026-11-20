"""Generate synthetic nadir (top-down) aerial frames for the demo scenarios.

Each image gets a ``.json`` sidecar with ground-truth perception, which the
``mock`` VLM backend returns and which you can use to evaluate a real VLM.

    python scripts/generate_samples.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples"
W, H = 1024, 768


def textured(base: tuple[int, int, int], jitter: int, seed: int, blur: float = 1.2) -> Image.Image:
    rnd = random.Random(seed)
    small = Image.new("RGB", (W // 4, H // 4))
    px = small.load()
    for y in range(H // 4):
        for x in range(W // 4):
            d = rnd.randint(-jitter, jitter)
            px[x, y] = tuple(max(0, min(255, c + d)) for c in base)
    return small.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(blur))


def shadowed_rect(d: ImageDraw.ImageDraw, box, fill, shadow=(8, 10), radius=4):
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0 + shadow[0], y0 + shadow[1], x1 + shadow[0], y1 + shadow[1]), radius, fill=(40, 45, 35))
    d.rounded_rectangle(box, radius, fill=fill, outline=(30, 30, 30))


def fence(d: ImageDraw.ImageDraw, box, gap=None):
    x0, y0, x1, y1 = box
    segs = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for (a, b) in segs:
        d.line([a, b], fill=(200, 200, 190), width=3)
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 24)
        for i in range(n + 1):
            px = a[0] + (b[0] - a[0]) * i / max(n, 1)
            py = a[1] + (b[1] - a[1]) * i / max(n, 1)
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(90, 90, 90))
    if gap:
        d.line(gap, fill=(120, 110, 80), width=6)


def car(d: ImageDraw.ImageDraw, cx, cy, color=(180, 30, 30), angle_vertical=False):
    w, h = (34, 70) if angle_vertical else (70, 34)
    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    shadowed_rect(d, box, color, radius=10)
    if angle_vertical:
        d.rounded_rectangle((box[0] + 5, box[1] + 10, box[2] - 5, box[1] + 24), 4, fill=(40, 60, 80))
        d.rounded_rectangle((box[0] + 5, box[3] - 20, box[2] - 5, box[3] - 10), 4, fill=(40, 60, 80))
    else:
        d.rounded_rectangle((box[2] - 24, box[1] + 5, box[2] - 10, box[3] - 5), 4, fill=(40, 60, 80))
        d.rounded_rectangle((box[0] + 10, box[1] + 5, box[0] + 20, box[3] - 5), 4, fill=(40, 60, 80))


def person(d: ImageDraw.ImageDraw, cx, cy, shirt=(30, 50, 120)):
    """Nadir view of a standing person (~0.5 m shoulders ≈ 24 px at 40 m AGL) with a long low-sun shadow."""
    d.polygon([(cx + 6, cy + 4), (cx + 58, cy + 30), (cx + 52, cy + 40), (cx - 2, cy + 12)], fill=(45, 45, 40))  # shadow
    d.ellipse((cx - 14, cy - 9, cx + 14, cy + 9), fill=shirt, outline=(15, 15, 20))  # shoulders/torso
    d.ellipse((cx - 6, cy - 7, cx - 1, cy + 7), fill=tuple(max(0, c - 25) for c in shirt))  # arm
    d.ellipse((cx + 1, cy - 7, cx + 6, cy + 7), fill=tuple(max(0, c - 25) for c in shirt))  # arm
    d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(55, 40, 30), outline=(20, 15, 10))  # head/hair


def transformer(d: ImageDraw.ImageDraw, x, y):
    shadowed_rect(d, (x, y, x + 90, y + 60), (120, 125, 120))
    for i in range(6):
        d.line((x + 8 + i * 14, y + 4, x + 8 + i * 14, y + 56), fill=(90, 95, 90), width=4)
    for i in range(3):
        d.ellipse((x + 15 + i * 25, y - 12, x + 29 + i * 25, y + 2), fill=(170, 120, 60), outline=(60, 40, 20))


def substation_base(seed: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = textured((92, 128, 70), 18, seed)
    d = ImageDraw.Draw(img)
    # access road
    d.rectangle((0, 610, W, 680), fill=(105, 105, 100))
    for x in range(0, W, 60):
        d.rectangle((x, 642, x + 30, 646), fill=(220, 210, 150))
    d.rectangle((640, 520, 700, 610), fill=(105, 105, 100))  # driveway to gate
    # gravel yard
    gravel = textured((160, 155, 145), 25, seed + 1, 0.6).crop((180, 90, 900, 520))
    img.paste(gravel, (180, 90))
    fence(d, (180, 90, 900, 520))
    d.rectangle((640, 516, 700, 524), fill=(105, 105, 100))  # gate opening
    # busbars and gantries
    for y in (150, 175, 200):
        d.line((200, y, 880, y), fill=(50, 50, 55), width=3)
    for x in (220, 400, 580, 760, 870):
        d.rectangle((x - 6, 140, x + 6, 210), outline=(70, 70, 70), width=2)
        d.line((x - 6, 140, x + 6, 210), fill=(70, 70, 70), width=1)
        d.line((x + 6, 140, x - 6, 210), fill=(70, 70, 70), width=1)
    # HV lines leaving the yard
    for dy in (-12, 0, 12):
        d.line((900, 175 + dy, W, 120 + dy), fill=(35, 35, 40), width=2)
    transformer(d, 260, 300)
    transformer(d, 420, 300)
    shadowed_rect(d, (600, 280, 760, 380), (200, 195, 185))  # control building
    d.line((600, 330, 760, 330), fill=(160, 155, 150), width=2)
    return img, d


def make_substation_vehicle() -> tuple[Image.Image, dict]:
    img, d = substation_base(7)
    car(d, 820, 470, color=(30, 30, 35))  # inside fence, near corner
    car(d, 120, 645, color=(200, 200, 205))  # legit car on public road
    truth = {
        "scene_summary": "Nadir view of an electrical substation with transformer yard, control building and perimeter fence; a dark vehicle is parked inside the fence near the south-east corner.",
        "anomaly": "unauthorized_vehicle_near_fence",
        "confidence": 0.91,
        "anomalies": [
            {"type": "unauthorized_vehicle_near_fence", "description": "Dark sedan parked inside the perimeter fence next to the south-east corner, away from the gate and driveway.", "confidence": 0.91, "severity": "high", "location_in_frame": "bottom-right"}
        ],
        "objects": [{"label": "car", "count": 2}, {"label": "transformer", "count": 2}, {"label": "building", "count": 1}],
        "aircraft_hazards": ["high_voltage_lines", "busbar_gantries"],
    }
    return img, truth


def make_substation_person() -> tuple[Image.Image, dict]:
    img, d = substation_base(11)
    # Cut west fence: erase a section and draw the mesh panel peeled outward.
    d.line((180, 370, 180, 440), fill=(92, 128, 70), width=8)
    d.polygon([(180, 370), (128, 382), (124, 432), (180, 440)], outline=(200, 200, 190), width=3)
    for y in range(378, 438, 9):
        d.line((180, y, 126, y + 4), fill=(170, 170, 160), width=1)
    d.rectangle((96, 395, 118, 425), fill=(60, 60, 65), outline=(20, 20, 20))  # bag/tool left outside
    person(d, 236, 402, shirt=(30, 50, 120))
    person(d, 318, 448, shirt=(140, 30, 30))
    truth = {
        "scene_summary": "Substation yard viewed from above; the west fence line appears breached and two people are inside the yard near the transformers.",
        "anomaly": "unauthorized_person_inside_fence",
        "confidence": 0.88,
        "anomalies": [
            {"type": "unauthorized_person_inside_fence", "description": "Two people inside the yard near transformer T1, close to a gap in the west fence.", "confidence": 0.88, "severity": "critical", "location_in_frame": "center-left"},
            {"type": "perimeter_fence_breach", "description": "Section of the west fence appears cut or open.", "confidence": 0.74, "severity": "high", "location_in_frame": "left"},
        ],
        "objects": [{"label": "person", "count": 2}, {"label": "transformer", "count": 2}],
        "aircraft_hazards": ["high_voltage_lines"],
    }
    return img, truth


def make_powerline_vegetation() -> tuple[Image.Image, dict]:
    img = textured((110, 135, 75), 22, 21)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 330, W, 450), fill=(140, 130, 95))  # corridor right-of-way
    for tx in (120, 900):
        d.polygon([(tx - 40, 390), (tx, 330), (tx + 40, 390), (tx, 450)], outline=(60, 60, 60), width=4)
        d.line((tx - 40, 390, tx + 40, 390), fill=(60, 60, 60), width=4)
        d.line((tx, 330, tx, 450), fill=(60, 60, 60), width=3)
    for dy, sag in ((-30, 22), (0, 30), (30, 22)):
        pts = [(x, 390 + dy + sag * (1 - ((x - 510) / 390) ** 2)) for x in range(120, 901, 10)]
        d.line(pts, fill=(25, 25, 30), width=3)
    rnd = random.Random(5)
    for _ in range(40):
        x, y = rnd.randint(0, W), rnd.choice([rnd.randint(40, 300), rnd.randint(480, 740)])
        r = rnd.randint(18, 36)
        d.ellipse((x - r + 6, y - r + 8, x + r + 6, y + r + 8), fill=(50, 70, 40))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(40 + rnd.randint(0, 30), 100 + rnd.randint(0, 40), 40))
    for x, y, r in ((520, 395, 44), (470, 372, 32), (565, 420, 30)):  # encroaching canopy
        d.ellipse((x - r + 6, y - r + 8, x + r + 6, y + r + 8), fill=(50, 70, 40))
        d.ellipse((x - r, y - r, x + r, y + r), fill=(45, 120, 45))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    truth = {
        "scene_summary": "Transmission-line corridor between two lattice towers; tree canopy at mid-span is growing into the conductors.",
        "anomaly": "vegetation_encroachment_on_conductor",
        "confidence": 0.84,
        "anomalies": [
            {"type": "vegetation_encroachment_on_conductor", "description": "Tree canopy at mid-span within ~2 m of the centre phase conductor.", "confidence": 0.84, "severity": "medium", "location_in_frame": "center"}
        ],
        "objects": [{"label": "transmission_tower", "count": 2}, {"label": "tree", "count": 43}],
        "aircraft_hazards": ["high_voltage_lines", "transmission_towers"],
    }
    return img, truth


def make_solar_damage() -> tuple[Image.Image, dict]:
    img = textured((150, 140, 110), 15, 31)
    d = ImageDraw.Draw(img)
    rnd = random.Random(9)
    damaged = {(2, 5), (2, 6), (4, 11)}
    for r in range(7):
        y = 60 + r * 98
        for c in range(14):
            x = 40 + c * 68
            box = (x, y, x + 62, y + 70)
            d.rectangle((x + 5, y + 8, x + 67, y + 78), fill=(90, 85, 70))
            col = (25, 45, 95) if (r, c) not in damaged else (60, 60, 70)
            d.rectangle(box, fill=col, outline=(180, 180, 190))
            for gx in range(1, 4):
                d.line((x + gx * 15.5, y, x + gx * 15.5, y + 70), fill=(70, 90, 140), width=1)
            for gy in range(1, 5):
                d.line((x, y + gy * 14, x + 62, y + gy * 14), fill=(70, 90, 140), width=1)
            if (r, c) in damaged:
                cx, cy = x + 31, y + 35
                for _ in range(7):
                    d.line((cx, cy, cx + rnd.randint(-30, 30), cy + rnd.randint(-34, 34)), fill=(210, 210, 215), width=2)
    truth = {
        "scene_summary": "Ground-mounted solar array of 7 rows; three modules show shattered glass / dark discolouration.",
        "anomaly": "damaged_solar_panel",
        "confidence": 0.86,
        "anomalies": [
            {"type": "damaged_solar_panel", "description": "Three modules (row 3 cols 6-7, row 5 col 12) with cracked glass.", "confidence": 0.86, "severity": "low", "location_in_frame": "center"}
        ],
        "objects": [{"label": "solar_panel", "count": 98}],
        "aircraft_hazards": [],
    }
    return img, truth


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    makers = {
        "substation_vehicle_fence": make_substation_vehicle,
        "substation_person_fence": make_substation_person,
        "powerline_vegetation": make_powerline_vegetation,
        "solar_panel_damage": make_solar_damage,
    }
    for name, fn in makers.items():
        img, truth = fn()
        img.save(OUT / f"{name}.png", optimize=True)
        (OUT / f"{name}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        print(f"wrote {OUT / name}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
