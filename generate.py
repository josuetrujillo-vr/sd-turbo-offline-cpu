"""Command-line image generation from local model files."""

from __future__ import annotations

from src.runtime import configure_logging, enable_offline

enable_offline()

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config, project_path, resolve_model

logger = logging.getLogger("generate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generacion local de imagenes en CPU/FP32.")
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--negative-prompt", "-n", help="Sin efecto en SD-Turbo con CFG=0.")
    parser.add_argument("--model", "-m", help="Alias o ruta configurada dentro de models/.")
    parser.add_argument("--profile", choices=["sfw", "nsfw"], default="sfw")
    parser.add_argument("--safety-checker", choices=["on", "off"])
    parser.add_argument("--lora", help="Archivo .safetensors dentro de models/.")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--embedding", help="Archivo .safetensors dentro de models/.")
    parser.add_argument("--steps", "-s", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--width", "-W", type=int)
    parser.add_argument("--height", "-H", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch", "-b", type=int, default=1, help="Imagenes secuenciales; seed incremental.")
    parser.add_argument("--output", "-o", help="Ruta relativa al proyecto o absoluta.")
    parser.add_argument("--backend", choices=["auto", "diffusers_cpu"], default="auto")
    parser.add_argument("--threads", type=int, help="Por defecto hardware.num_threads.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def save_image_with_metadata(image, output_path: Path, prompt: str,
                             negative_prompt: str, model_name: str,
                             seed: int | None, metrics: dict) -> Path:
    from PIL.PngImagePlugin import PngInfo

    pnginfo = PngInfo()
    for key, value in {
        "prompt": prompt, "negative_prompt": negative_prompt, "model": model_name,
        "seed": seed, "generated_by": "ai-text-to-image-cpu",
        "generation_date": datetime.now().isoformat(),
        "metrics": json.dumps(metrics, ensure_ascii=False),
    }.items():
        pnginfo.add_text(key, str(value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "PNG", pnginfo=pnginfo)
    logger.info("Imagen guardada: %s", output_path)
    return output_path


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        from src.pipeline import (GenerationConfig, apply_lora, configure_threads,
                                  generate_image, load_diffusers_pipeline,
                                  load_textual_inversion)

        config = load_config()
        profile = config["profiles"][args.profile]
        model = args.model or profile["model"]
        _, spec = resolve_model(model, config)
        if args.batch < 1:
            raise ValueError("batch debe ser positivo.")
        if args.seed is not None and args.seed + args.batch - 1 >= 2**63:
            raise ValueError("Las semillas del lote exceden 2**63-1.")
        gen = GenerationConfig(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt is not None else profile["negative_prompt"],
            num_inference_steps=args.steps if args.steps is not None else spec["steps"],
            guidance_scale=args.cfg if args.cfg is not None else spec["guidance_scale"],
            width=args.width if args.width is not None else config["generation"]["width"],
            height=args.height if args.height is not None else config["generation"]["height"],
            seed=args.seed,
        )
        configure_threads(args.threads)
        safety = profile["safety_checker"] if args.safety_checker is None else args.safety_checker == "on"
        pipe = load_diffusers_pipeline(model, safety_checker=safety)
        if args.lora:
            apply_lora(pipe, args.lora, args.lora_scale)
        if args.embedding:
            load_textual_inversion(pipe, args.embedding)

        output_dir = project_path(args.output or profile["output_dir"])
        for index in range(args.batch):
            gen.seed = args.seed + index if args.seed is not None else None
            image, metrics = generate_image(pipe, gen)
            filename = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{index:03d}_{metrics['seed']}.png"
            save_image_with_metadata(image, output_dir / filename, gen.prompt,
                                     gen.negative_prompt, metrics["model"], metrics["seed"], metrics)
        logger.info("Generacion completada: %d imagenes en %s", args.batch, output_dir)
        return 0
    except Exception:
        logger.exception("Fallo la generacion")
        return 1


if __name__ == "__main__":
    sys.exit(main())
