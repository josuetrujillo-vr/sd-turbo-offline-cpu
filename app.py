"""Gradio UI on loopback, with local assets and serialized CPU inference."""

from __future__ import annotations

from src.runtime import configure_logging, enable_offline, plain_text

enable_offline()  # Gradio reads analytics settings during import.

import gc
import logging
import threading
from datetime import datetime

from src.config import load_config, model_path, project_path, resolve_model

logger = logging.getLogger("app")
_pipe_cache = {"key": None, "pipe": None}
# ponytail: one shared CPU pipeline; serialize both UI events and direct calls.
_generation_lock = threading.Lock()


def get_or_load_pipeline(model_id: str, backend: str, safety_checker: bool,
                         lora_file: str | None = None, lora_scale: float = 1.0):
    from src.pipeline import apply_lora, configure_threads, load_diffusers_pipeline

    if backend not in ("auto", "diffusers_cpu"):
        raise ValueError("Solo se admite diffusers_cpu.")
    config = load_config()
    path, _ = resolve_model(model_id, config)
    lora_path = model_path(lora_file) if lora_file else None
    lora_key = (str(lora_path), lora_path.stat().st_mtime_ns, lora_scale) if lora_path else None
    key = (str(path), safety_checker, lora_key, repr(config["vae"]),
           repr(config["hardware"]), repr(config["safety_checker"]))
    if _pipe_cache["pipe"] is None or _pipe_cache["key"] != key:
        # Reset before loading so a failed reload cannot leave stale cache keys.
        _pipe_cache.update(key=None, pipe=None)
        gc.collect()
        configure_threads()
        pipe = load_diffusers_pipeline(model_id, safety_checker=safety_checker)
        if lora_path:
            apply_lora(pipe, str(lora_path), lora_scale)
        _pipe_cache.update(key=key, pipe=pipe)
    return _pipe_cache["pipe"]


def generate(prompt: str, negative_prompt: str, model_name: str, profile: str,
             steps: int, cfg: float, width: int, height: int, seed_value: int,
             use_seed: bool, lora_file: str | None, lora_scale: float) -> tuple:
    try:
        with _generation_lock:
            from generate import save_image_with_metadata
            from src.pipeline import GenerationConfig, generate_image

            config = load_config()
            profile_data = config["profiles"][profile]
            model_name = model_name or profile_data["model"]
            gen = GenerationConfig(
                prompt=prompt, negative_prompt=negative_prompt or profile_data["negative_prompt"],
                num_inference_steps=steps, guidance_scale=cfg, width=width, height=height,
                seed=seed_value if use_seed else None,
            )
            pipe = get_or_load_pipeline(model_name, "diffusers_cpu",
                                         profile_data["safety_checker"], lora_file, lora_scale)
            image, metrics = generate_image(pipe, gen)
            output_dir = project_path(profile_data["output_dir"])
            filepath = output_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{metrics['seed']}.png"
            save_image_with_metadata(image, filepath, gen.prompt, gen.negative_prompt,
                                     metrics["model"], metrics["seed"], metrics)
            info = (
                f"Modelo: {metrics['model']}\nBackend: diffusers_cpu\n"
                f"Resolucion: {metrics['resolution']}\nPasos: {steps}; CFG: {cfg}\n"
                f"Seed: {metrics['seed']}\nTiempo: {metrics['total_seconds']:.1f}s\n"
                f"RSS final: {metrics['ram_after_mb']:.0f} MiB\nGuardado: {filepath}"
            )
            gallery = [str(p) for p in sorted(output_dir.glob("*.png"),
                       key=lambda p: p.stat().st_mtime_ns, reverse=True)[:16]]
            return image, info, gallery
    except Exception as exc:
        logger.exception("Fallo la generacion en Gradio (perfil=%s)", profile)
        return None, plain_text(f"{type(exc).__name__}: {exc}"), []


def build_ui():
    """Build without launching, so schema and callbacks can be tested."""
    import gradio as gr

    config = load_config()
    default_model = config["profiles"]["sfw"]["model"]
    spec = config["models"][default_model]
    theme = gr.themes.Soft(font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"])
    with gr.Blocks(title="Text-to-Image CPU", theme=theme, analytics_enabled=False) as demo:
        gr.Markdown("# Generador de imagenes local\nCPU / FP32. Modelos preparados con download_model.py.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", lines=3)
                negative = gr.Textbox(label="Negative prompt (sin efecto con SD-Turbo CFG=0)", lines=2)
                profile = gr.Radio(["sfw", "nsfw"], value="sfw", label="Perfil")
                model = gr.Dropdown(list(config["models"]), value=default_model, label="Modelo local")
                with gr.Accordion("Parametros avanzados", open=False):
                    steps = gr.Slider(1, 50, value=spec["steps"], step=1, label="Pasos")
                    cfg = gr.Slider(0, 20, value=spec["guidance_scale"], step=0.5,
                                    label="CFG (SD-Turbo: 0)")
                    width = gr.Slider(64, 1024, value=config["generation"]["width"], step=8, label="Ancho")
                    height = gr.Slider(64, 1024, value=config["generation"]["height"], step=8, label="Alto")
                    use_seed = gr.Checkbox(label="Usar seed", value=False)
                    seed = gr.Number(label="Seed", value=42, precision=0)
                    lora = gr.Textbox(label="LoRA local (.safetensors dentro de models/)")
                    lora_scale = gr.Slider(0, 2, value=1, step=0.05, label="Escala LoRA")
                button = gr.Button("Generar imagen", variant="primary")
            with gr.Column():
                result = gr.Image(type="pil", label="Imagen generada")
                info = gr.Textbox(label="Resultado o error", lines=8, interactive=False)
                gallery = gr.Gallery(label="Resultados recientes", columns=4, rows=2)
        inputs = [prompt, negative, model, profile, steps, cfg, width, height, seed, use_seed, lora, lora_scale]
        outputs = [result, info, gallery]
        button.click(generate, inputs, outputs, concurrency_id="generation", concurrency_limit=1)
        prompt.submit(generate, inputs, outputs, concurrency_id="generation", concurrency_limit=1)
    return demo.queue(default_concurrency_limit=1)


def main() -> int:
    configure_logging()
    try:
        config = load_config()
        demo = build_ui()
        demo.launch(server_name="127.0.0.1", server_port=config["web"]["port"],
                    share=False, inbrowser=False, show_error=True)
        return 0
    except Exception:
        logger.exception("Fallo el inicio de Gradio")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
