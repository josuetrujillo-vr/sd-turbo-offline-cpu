"""One-time online preparation. Inference never imports this script."""

import argparse
import logging
import os
import subprocess
import sys

from src.config import PROJECT_ROOT, load_config, model_path
from src.runtime import configure_logging

logger = logging.getLogger("download_model")

MODEL_FILES = [
    "model_index.json", "scheduler/scheduler_config.json",
    "tokenizer/merges.txt", "tokenizer/vocab.json",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]
VAE_FILES = ["config.json", "diffusion_pytorch_model.safetensors"]
SAFETY_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga inicial de modelos locales FP32.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Descargar nuevamente los archivos.")
    mode.add_argument("--verify-only", action="store_true", help="Verificar carga local sin red.")
    args = parser.parse_args()
    configure_logging()
    try:
        config = load_config()
        if not args.verify_only:
            if sys.platform == "win32":
                # Use Windows' trusted certificates; keep TLS verification enabled.
                import truststore
                truststore.inject_into_ssl()
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ[key] = "0"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            from huggingface_hub import snapshot_download

            resources = [(s, MODEL_FILES) for s in config["models"].values()]
            resources += [(config["vae"], VAE_FILES), (config["safety_checker"], SAFETY_FILES)]
            for spec, files in resources:
                path = model_path(spec["path"], must_exist=False)
                logger.info("Preparando %s en %s", spec["repo"], path)
                snapshot_download(
                    repo_id=spec["repo"], revision=spec["revision"],
                    local_dir=str(path),
                    allow_patterns=files + ["README.md", "LICENSE*"],
                    force_download=args.force, max_workers=2,
                )
                missing = [name for name in files if not (path / name).is_file()]
                if missing:
                    raise FileNotFoundError(f"Descarga incompleta en {path}: {missing}")

        # A new process reads Hub constants with offline flags enabled.
        subprocess.run([
            sys.executable, "-c",
            "from src.pipeline import configure_threads, load_diffusers_pipeline; "
            "configure_threads(); "
            "pipe = load_diffusers_pipeline('sd_turbo', safety_checker=True); "
            "print('Verificacion local de SD-Turbo, VAE y safety checker: OK')",
        ], cwd=PROJECT_ROOT, check=True)
        logger.info("Preparacion completada. Ya puedes generar sin internet.")
        return 0
    except Exception:
        logger.exception("Fallo la preparacion del modelo")
        return 1


if __name__ == "__main__":
    sys.exit(main())
