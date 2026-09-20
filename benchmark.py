"""Warm CPU benchmarks using prepared local models; no implicit downloads."""

from src.runtime import configure_logging, enable_offline

enable_offline()

import argparse
import json
import logging
import platform
from datetime import datetime

from src.config import load_config, project_path

logger = logging.getLogger("benchmark")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local CPU: warmup y 3 repeticiones.")
    parser.add_argument("--quick", action="store_true", help="Solo 512x512, un paso, primer modelo.")
    parser.add_argument("--model", help="Alias en config.yaml.")
    parser.add_argument("--output", "-o", default="benchmark_results")
    args = parser.parse_args()
    configure_logging()
    try:
        from src.pipeline import (GenerationConfig, clear_memory, configure_threads,
                                  generate_image, load_diffusers_pipeline)
        import psutil
        import torch

        config = load_config()
        models = config["models"]
        if args.model:
            models = {args.model: models[args.model]}
        elif args.quick:
            name = next(iter(models))
            models = {name: models[name]}
        configure_threads()
        runs = []
        failed = False
        for name, spec in models.items():
            pipe = None
            try:
                pipe = load_diffusers_pipeline(name, safety_checker=False)
                for size in ([512] if args.quick else [256, 384, 512]):
                    for steps in ([1] if args.quick else [1, 4]):
                        gen = GenerationConfig("a photograph of an astronaut riding a horse on mars",
                                               width=size, height=size, num_inference_steps=steps,
                                               guidance_scale=spec["guidance_scale"], seed=42)
                        generate_image(pipe, gen)
                        samples = []
                        for seed in (43, 44, 45):
                            gen.seed = seed
                            _, metrics = generate_image(pipe, gen)
                            samples.append(metrics)
                        runs.append({"model": name, "resolution": f"{size}x{size}", "steps": steps,
                                     "seconds_avg": sum(x["total_seconds"] for x in samples) / 3,
                                     "rss_after_mb_max": max(x["ram_after_mb"] for x in samples),
                                     "samples": samples})
            except Exception as exc:
                logger.exception("Fallo el benchmark de %s", name)
                runs.append({"model": name, "error": str(exc)})
                failed = True
            finally:
                del pipe
                clear_memory()
        report = {"date": datetime.now().isoformat(), "platform": platform.platform(),
                  "python": platform.python_version(), "torch": torch.__version__,
                  "backend": "diffusers_cpu", "threads": torch.get_num_threads(),
                  "ram_total_gib": psutil.virtual_memory().total / 1024**3, "runs": runs}
        output = project_path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (output / f"benchmark_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = ["# Benchmark CPU", "", "Una imagen de warmup y tres imagenes medidas por configuracion.",
                 "RSS al finalizar cada imagen; no mide el pico de memoria.", "",
                 "| Modelo | Resolucion | Pasos | Media (s) | RSS final max. (MiB) |",
                 "|---|---|---|---|---|"]
        for run in runs:
            if "error" in run:
                lines.append(f"| {run['model']} | ERROR | | | |")
            else:
                lines.append(f"| {run['model']} | {run['resolution']} | {run['steps']} | "
                             f"{run['seconds_avg']:.2f} | {run['rss_after_mb_max']:.0f} |")
        (output / f"benchmark_{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
        logger.info("Resultados guardados en %s", output)
        return int(failed)
    except Exception:
        logger.exception("Fallo el benchmark")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
