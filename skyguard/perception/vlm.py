"""Vision-language model backends + tiled (SAHI-style) inference.

All backends share one prompt and one output schema (``PerceptionResult``).

* ``groq``       – hosted vision model on Groq (default: qwen/qwen3.8-27b, needs GROQ_API_KEY).
* ``qwen_local`` – Qwen2.5-VL-3B-Instruct, 4-bit NF4 via bitsandbytes, fits a 4 GB GPU.
* ``hf_api``     – Hugging Face Inference Providers (needs HF_TOKEN with Inference permission).
* ``mock``       – deterministic, reads ``<image>.json`` ground truth. Offline demo/tests.

Small-object problem: from 40 m AGL a person is ~25 px in a 1024 px nadir frame,
below what VLM vision encoders resolve after their internal down-sampling. With
tiling enabled we analyse the full frame *plus* overlapping zoomed tiles in
parallel and merge detections (same idea as SAHI for detectors).
"""

from __future__ import annotations

import base64
import io
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import Settings
from .schema import PERCEPTION_PROMPT, Anomaly, PerceptionResult, parse_perception

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
TILE_PROMPT_SUFFIX = (
    "\n\nNOTE: this image is a ZOOMED-IN TILE covering the {region} part of the full frame. "
    "Objects look larger than in the full frame. Report anomalies visible in this tile only."
)


def load_image(path: str | Path, max_side: int = 1280) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def image_to_data_url(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_prompt(telemetry: dict[str, Any]) -> str:
    sector = telemetry.get("sector", {}) or {}
    return PERCEPTION_PROMPT.format(
        sector=sector.get("name", telemetry.get("sector_name", "unknown")),
        alt_m=float(telemetry.get("alt_m", 0)),
        near_hv=bool(sector.get("near_hv_lines", False)),
    )


def make_tiles(img: Image.Image, grid: int = 2, overlap: float = 0.2, out_side: int = 1024) -> list[tuple[str, Image.Image]]:
    """Split into grid x grid overlapping tiles, each upscaled to ``out_side`` on its long edge."""
    w, h = img.size
    tw, th = int(w / grid * (1 + overlap)), int(h / grid * (1 + overlap))
    names_y = ["top", "bottom"] if grid == 2 else [f"row{i}" for i in range(grid)]
    names_x = ["left", "right"] if grid == 2 else [f"col{i}" for i in range(grid)]
    tiles = []
    for iy in range(grid):
        for ix in range(grid):
            x0 = min(int(ix * (w - tw) / max(grid - 1, 1)), w - tw)
            y0 = min(int(iy * (h - th) / max(grid - 1, 1)), h - th)
            tile = img.crop((x0, y0, x0 + tw, y0 + th))
            scale = out_side / max(tile.size)
            tile = tile.resize((int(tile.width * scale), int(tile.height * scale)), Image.LANCZOS)
            tiles.append((f"{names_y[iy]}-{names_x[ix]}", tile))
    return tiles


def merge_results(full: PerceptionResult, tiles: list[tuple[str, PerceptionResult]], min_conf: float = 0.5) -> PerceptionResult:
    """Union anomalies across full frame + tiles; keep best confidence per anomaly type."""
    best: dict[str, Anomaly] = {a.type: a for a in full.anomalies if a.type != "none"}
    found_in_tiles: list[str] = []
    for region, res in tiles:
        for a in res.anomalies:
            if a.type in {"none", "unknown"} or a.confidence < min_conf:
                continue
            a = a.model_copy(update={"location_in_frame": region if a.location_in_frame in {"", "unknown"} else f"{region} ({a.location_in_frame})"})
            if a.type not in best or a.confidence > best[a.type].confidence:
                best[a.type] = a
                found_in_tiles.append(f"{a.type}@{region}")
    merged = full.model_copy(deep=True)
    merged.anomalies = sorted(best.values(), key=lambda a: (SEVERITY_RANK[a.severity], a.confidence), reverse=True)
    if merged.anomalies:
        merged.anomaly, merged.confidence = merged.anomalies[0].type, merged.anomalies[0].confidence
    else:
        merged.anomaly = "none"
    merged.aircraft_hazards = sorted({*full.aircraft_hazards, *(h for _, r in tiles for h in r.aircraft_hazards)})
    if found_in_tiles:
        merged.scene_summary = f"{full.scene_summary} [Tiled pass added: {', '.join(found_in_tiles)}]"
    return merged


class VLMBackend(ABC):
    name = "base"
    model = ""
    max_side = 1024

    def __init__(self, tiling: bool = False, tile_grid: int = 2):
        self.tiling = tiling
        self.tile_grid = tile_grid

    def analyze(self, image_path: str | Path, telemetry: dict[str, Any]) -> PerceptionResult:
        started = time.perf_counter()
        img = load_image(image_path, self.max_side)
        prompt = build_prompt(telemetry)
        if not self.tiling:
            result = self._infer(img, prompt)
            result.tiles_analyzed = 0
        else:
            tiles = make_tiles(img, self.tile_grid, out_side=self.max_side)
            jobs = [("full", img, prompt)] + [(r, t, prompt + TILE_PROMPT_SUFFIX.format(region=r)) for r, t in tiles]
            with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
                outs = list(pool.map(lambda j: (j[0], self._safe_infer(j[1], j[2])), jobs))
            full = outs[0][1]
            if full is None:
                raise RuntimeError(f"Full-frame VLM call failed: {self._last_error}")
            result = merge_results(full, [(r, o) for r, o in outs[1:] if o is not None])
            result.tiles_analyzed = sum(1 for _, o in outs[1:] if o is not None)
        result.backend, result.model = self.name, self.model
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    parallelism = 5  # API backends run tiles concurrently; local GPU overrides to 1

    _last_error = ""

    def _safe_infer(self, img: Image.Image, prompt: str) -> PerceptionResult | None:
        try:
            return self._infer(img, prompt)
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            return None

    @abstractmethod
    def _infer(self, img: Image.Image, prompt: str) -> PerceptionResult: ...


# ----------------------------------------------------------------------- Groq
class GroqVLM(VLMBackend):
    name = "groq"

    def __init__(self, api_key: str, model: str, **kw):
        super().__init__(**kw)
        from groq import Groq

        # Free tier is ~8k tokens/min/model: retry 429s (SDK honours retry-after) instead of failing.
        self.client = Groq(api_key=api_key, max_retries=5)
        self.model = model

    def _infer(self, img: Image.Image, prompt: str) -> PerceptionResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=700,  # JSON answer is ~250-400 tokens; reserving less helps rate limits
            response_format={"type": "json_object"},
        )
        return parse_perception(resp.choices[0].message.content or "{}")


# ------------------------------------------------------------- Qwen2.5-VL local
class QwenLocalVLM(VLMBackend):
    """Qwen2.5-VL-3B in 4-bit. Peak VRAM ~3.2 GB with visual tokens capped at 640."""

    name = "qwen_local"
    max_side = 896
    parallelism = 1  # one GPU: run tiles sequentially
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 640 * 28 * 28

    def __init__(self, model: str, **kw):
        super().__init__(**kw)
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "qwen_local needs a CUDA build of PyTorch: "
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
            )
        self.model = model
        self.torch = torch
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, quantization_config=bnb, device_map="cuda:0", dtype=torch.float16
        )
        self.vlm.eval()
        # Cap visual tokens so the 4 GB RTX 3050 Ti does not OOM.
        self.processor = AutoProcessor.from_pretrained(model)
        ip = self.processor.image_processor
        ip.size = {"shortest_edge": self.MIN_PIXELS, "longest_edge": self.MAX_PIXELS}
        for attr, val in (("min_pixels", self.MIN_PIXELS), ("max_pixels", self.MAX_PIXELS)):
            if hasattr(ip, attr):
                setattr(ip, attr, val)

    def _infer(self, img: Image.Image, prompt: str) -> PerceptionResult:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.vlm.device)
        with self.torch.inference_mode():
            out = self.vlm.generate(**inputs, max_new_tokens=512, do_sample=False)
        decoded = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        return parse_perception(decoded)


# ---------------------------------------------------------------- HF Inference
class HFApiVLM(VLMBackend):
    name = "hf_api"

    def __init__(self, token: str, model: str, **kw):
        super().__init__(**kw)
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(api_key=token)
        self.model = model

    def _infer(self, img: Image.Image, prompt: str) -> PerceptionResult:
        resp = self.client.chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return parse_perception(resp.choices[0].message.content or "{}")


# ------------------------------------------------------------------------ Mock
class MockVLM(VLMBackend):
    """Deterministic stand-in. Uses ``<image>.json`` ground truth if present."""

    name = "mock"
    model = "ground-truth-sidecar"

    KEYWORDS = {
        "vehicle": ("unauthorized_vehicle_near_fence", 0.91, "high"),
        "person": ("unauthorized_person_inside_fence", 0.88, "critical"),
        "fire": ("smoke_or_fire", 0.9, "critical"),
        "vegetation": ("vegetation_encroachment_on_conductor", 0.84, "medium"),
        "solar": ("damaged_solar_panel", 0.86, "low"),
    }

    def analyze(self, image_path: str | Path, telemetry: dict[str, Any]) -> PerceptionResult:
        started = time.perf_counter()
        image_path = Path(image_path)
        result = self._from_path(image_path)
        result.backend, result.model = self.name, self.model
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    def _infer(self, img: Image.Image, prompt: str) -> PerceptionResult:  # pragma: no cover - unused
        raise NotImplementedError

    def _from_path(self, image_path: Path) -> PerceptionResult:
        sidecar = image_path.with_suffix(".json")
        if sidecar.exists():
            return parse_perception(sidecar.read_text(encoding="utf-8"))
        stem = image_path.stem.lower()
        for kw, (atype, conf, sev) in self.KEYWORDS.items():
            if kw in stem:
                payload = {
                    "scene_summary": f"Mock perception keyed on filename '{image_path.name}'.",
                    "anomalies": [{"type": atype, "confidence": conf, "severity": sev, "description": atype.replace("_", " ")}],
                }
                return parse_perception(json.dumps(payload))
        return parse_perception(json.dumps({"scene_summary": "No anomaly (mock).", "anomaly": "none", "confidence": 0.0}))


def build_vlm(settings: Settings, backend: str | None = None) -> VLMBackend:
    backend = backend or settings.resolved_vlm_backend()
    tiling = settings.vlm_tiling_for(backend)
    if backend == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("VLM_BACKEND=groq but GROQ_API_KEY is not set in .env")
        return GroqVLM(settings.groq_api_key, settings.groq_vlm_model, tiling=tiling)
    if backend == "qwen_local":
        return QwenLocalVLM(settings.qwen_local_model, tiling=tiling)
    if backend == "hf_api":
        if not settings.hf_token:
            raise RuntimeError("VLM_BACKEND=hf_api but HF_TOKEN is not set in .env")
        return HFApiVLM(settings.hf_token, settings.hf_vlm_model, tiling=tiling)
    if backend == "mock":
        return MockVLM()
    raise ValueError(f"Unknown VLM backend '{backend}'")
