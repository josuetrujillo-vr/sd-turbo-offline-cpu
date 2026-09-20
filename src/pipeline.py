"""Local Diffusers inference on CPU/FP32. No network fallback."""

from __future__ import annotations

from src.runtime import enable_offline

enable_offline()  # Must precede torch, transformers and diffusers imports.

import gc
import logging
import math
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

import psutil
import torch

from src.config import load_config, model_path, resolve_model
from src.guardrails import PromptGuardrail

logger = logging.getLogger(__name__)
_interop_configured = False


def configure_threads(num_threads: int | None = None) -> None:
    global _interop_configured
    if num_threads is None:
        num_threads = load_config()["hardware"]["num_threads"]
    if type(num_threads) is not int or num_threads < 1:
        raise ValueError("threads debe ser un entero positivo.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(num_threads)
    torch.set_num_threads(num_threads)
    if not _interop_configured:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            logger.warning("PyTorch ya inicio trabajo paralelo; se conservan los hilos inter-op.")
        _interop_configured = True


@dataclass
class GenerationConfig:
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 1
    guidance_scale: float = 0.0
    width: int = 512
    height: int = 512
    seed: int | None = None
    num_images: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("El prompt no puede estar vacio.")
        if type(self.num_inference_steps) is not int or not 1 <= self.num_inference_steps <= 50:
            raise ValueError("steps debe ser un entero entre 1 y 50.")
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 64 <= value <= 1024 or value % 8:
                raise ValueError(f"{name} debe ser multiplo de 8 entre 64 y 1024.")
        if not math.isfinite(self.guidance_scale) or not 0 <= self.guidance_scale <= 20:
            raise ValueError("cfg debe estar entre 0 y 20.")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed debe ser un entero entre 0 y 2**63-1.")
        if self.num_images != 1:
            raise ValueError("generate_image genera una imagen; usa --batch para una secuencia.")
        result = PromptGuardrail().evaluate(self.prompt)
        if not result.allowed:
            raise ValueError(f"{result.reason} {result.details or ''}")


def detect_backend() -> str:
    return "diffusers_cpu"


def get_ram_usage_mb() -> float:
    """Current process RSS in MiB, not a peak measurement."""
    return psutil.Process().memory_info().rss / 1024**2


def clear_memory() -> None:
    gc.collect()


def load_diffusers_pipeline(
    model_id: str = "sd_turbo", safety_checker: bool = False,
    device: str = "cpu", dtype: torch.dtype = torch.float32,
) -> Any:
    if device != "cpu" or dtype != torch.float32:
        raise ValueError("Este proyecto ejecuta exclusivamente en CPU con torch.float32.")
    config = load_config()
    path, spec = resolve_model(model_id, config)
    if not (path / "model_index.json").is_file():
        raise FileNotFoundError(f"Falta model_index.json en {path}. Ejecuta python download_model.py.")

    from diffusers import AutoencoderTiny, StableDiffusionPipeline

    kwargs = dict(torch_dtype=dtype, local_files_only=True, use_safetensors=True,
                  safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    if config["vae"]["enabled"]:
        # Supply Tiny VAE at construction: avoid allocating the original VAE first.
        kwargs["vae"] = AutoencoderTiny.from_pretrained(
            str(model_path(config["vae"]["path"])), torch_dtype=dtype,
            local_files_only=True, use_safetensors=True,
        )
    if safety_checker:
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        safety_path = str(model_path(config["safety_checker"]["path"]))
        kwargs["safety_checker"] = StableDiffusionSafetyChecker.from_pretrained(
            safety_path, torch_dtype=dtype, local_files_only=True,
        )
        kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            safety_path, local_files_only=True,
        )
        kwargs["requires_safety_checker"] = True

    start = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(str(path), **kwargs).to("cpu")
    if config["vae"]["tiling"]:
        pipe.vae.enable_tiling()
    if config["hardware"]["attention_slicing"]:
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(ascii=True)
    pipe._model_name = spec["repo"]
    pipe._is_turbo = spec["repo"] == "stabilityai/sd-turbo"
    logger.info("Modelo local cargado en %.1fs; RSS actual: %.0f MiB",
                time.perf_counter() - start, get_ram_usage_mb())
    return pipe


def apply_lora(pipe: Any, lora_path: str, scale: float = 1.0) -> Any:
    path = model_path(lora_path)
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError("LoRA debe ser un archivo .safetensors local dentro de models/.")
    if not math.isfinite(scale) or not 0 <= scale <= 2:
        raise ValueError("lora-scale debe estar entre 0 y 2.")
    pipe.load_lora_weights(str(path.parent), weight_name=path.name,
                           local_files_only=True, use_safetensors=True,
                           adapter_name="local")
    # No fusion: changing/removing an adapter must not modify base weights.
    pipe.set_adapters(["local"], adapter_weights=[scale])
    return pipe


def load_textual_inversion(pipe: Any, embedding_path: str) -> Any:
    path = model_path(embedding_path)
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError("Embedding debe ser un .safetensors local dentro de models/.")
    pipe.load_textual_inversion(str(path.parent), weight_name=path.name,
                                local_files_only=True, use_safetensors=True)
    return pipe


def generate_image(pipe: Any, config: GenerationConfig,
                   backend: str = "diffusers_cpu") -> tuple[Any, dict]:
    if backend != "diffusers_cpu":
        raise ValueError("Solo se admite el backend diffusers_cpu.")
    config.__post_init__()  # Callers such as benchmark may change the dataclass.
    if getattr(pipe, "_is_turbo", False) and config.guidance_scale != 0:
        raise ValueError("SD-Turbo requiere cfg=0.0 en este proyecto.")
    if config.guidance_scale <= 1 and config.negative_prompt:
        logger.warning("negative_prompt no tiene efecto con CFG <= 1.")
    seed = config.seed if config.seed is not None else secrets.randbelow(2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ram_before = get_ram_usage_mb()
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=config.prompt, negative_prompt=config.negative_prompt or None,
            num_inference_steps=config.num_inference_steps, guidance_scale=config.guidance_scale,
            width=config.width, height=config.height, generator=generator,
            num_images_per_prompt=1,
        )
    elapsed = time.perf_counter() - start
    if getattr(result, "nsfw_content_detected", None) is not None and any(result.nsfw_content_detected):
        raise ValueError("El safety checker marco la imagen como NSFW; no se guardo la imagen filtrada.")
    if not result.images:
        raise RuntimeError("El pipeline no devolvio imagenes.")
    metrics = {
        "model": getattr(pipe, "_model_name", "unknown"), "backend": backend,
        "resolution": f"{config.width}x{config.height}", "steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale, "seed": seed,
        "total_seconds": round(elapsed, 3),
        "seconds_per_step": round(elapsed / config.num_inference_steps, 3),
        "ram_before_mb": round(ram_before, 1),
        "ram_after_mb": round(get_ram_usage_mb(), 1),
    }
    logger.info("Imagen %s, seed=%d, %.1fs; RSS final: %.0f MiB",
                metrics["resolution"], seed, elapsed, metrics["ram_after_mb"])
    return result.images[0], metrics
