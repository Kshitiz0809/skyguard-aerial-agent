"""Vision-language model backends.

All backends share one prompt and one output schema (``PerceptionResult``).

* ``groq``       – Llama-4-Scout vision on Groq's free tier (needs GROQ_API_KEY). Default.
* ``qwen_local`` – Qwen2.5-VL-3B-Instruct, 4-bit NF4 via bitsandbytes, fits a 4 GB GPU.
* ``hf_api``     – Hugging Face Inference Providers (needs HF_TOKEN).
* ``mock``       – deterministic, reads ``<image>.json`` ground truth. Offline demo/tests.
"""

from __future__ import annotations

import base64
import io
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import Settings
from .schema import PERCEPTION_PROMPT, PerceptionResult, parse_perception


def load_image(path: str | Path, max_side: int = 1280) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def image_to_data_url(img: Image.Image, quality: int = 88) -> str:
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


class VLMBackend(ABC):
    name = "base"
    model = ""

    def analyze(self, image_path: str | Path, telemetry: dict[str, Any]) -> PerceptionResult:
        started = time.perf_counter()
        result = self._analyze(Path(image_path), telemetry)
        result.backend, result.model = self.name, self.model
        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return result

    @abstractmethod
    def _analyze(self, image_path: Path, telemetry: dict[str, Any]) -> PerceptionResult: ...


# ----------------------------------------------------------------------- Groq
class GroqVLM(VLMBackend):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def _analyze(self, image_path: Path, telemetry: dict[str, Any]) -> PerceptionResult:
        img = load_image(image_path, 1024)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt(telemetry)},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )
        return parse_perception(resp.choices[0].message.content or "{}")


# ------------------------------------------------------------- Qwen2.5-VL local
class QwenLocalVLM(VLMBackend):
    """Qwen2.5-VL-3B in 4-bit. Peak VRAM ~3.2 GB with max_pixels capped at 640*28*28."""

    name = "qwen_local"

    def __init__(self, model: str):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "qwen_local needs a CUDA build of PyTorch. Install with: "
                "pip install -r requirements-gpu.txt (see README 'Local GPU VLM')."
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
            model, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
        )
        # Cap visual tokens so the 4 GB RTX 3050 Ti does not OOM.
        self.processor = AutoProcessor.from_pretrained(model, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)

    def _analyze(self, image_path: Path, telemetry: dict[str, Any]) -> PerceptionResult:
        img = load_image(image_path, 896)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(telemetry)}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.vlm.device)
        with self.torch.inference_mode():
            out = self.vlm.generate(**inputs, max_new_tokens=512, do_sample=False)
        decoded = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        return parse_perception(decoded)


# ---------------------------------------------------------------- HF Inference
class HFApiVLM(VLMBackend):
    name = "hf_api"

    def __init__(self, token: str, model: str):
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(api_key=token)
        self.model = model

    def _analyze(self, image_path: Path, telemetry: dict[str, Any]) -> PerceptionResult:
        img = load_image(image_path, 1024)
        resp = self.client.chat_completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                        {"type": "text", "text": build_prompt(telemetry)},
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

    def _analyze(self, image_path: Path, telemetry: dict[str, Any]) -> PerceptionResult:
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
    if backend == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("VLM_BACKEND=groq but GROQ_API_KEY is not set in .env")
        return GroqVLM(settings.groq_api_key, settings.groq_vlm_model)
    if backend == "qwen_local":
        return QwenLocalVLM(settings.qwen_local_model)
    if backend == "hf_api":
        if not settings.hf_token:
            raise RuntimeError("VLM_BACKEND=hf_api but HF_TOKEN is not set in .env")
        return HFApiVLM(settings.hf_token, settings.hf_vlm_model)
    if backend == "mock":
        return MockVLM()
    raise ValueError(f"Unknown VLM backend '{backend}'")
