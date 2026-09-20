"""Optional local ONNX utilities. Not used by Gradio or the generation CLI.

Install requirements-onnx.txt in a separate environment before exporting.
Export/quantization are experimental and do not claim a speed or quality gain.
"""

from src.runtime import enable_offline

enable_offline()

from dataclasses import dataclass
from pathlib import Path
import time

from src.config import model_path, resolve_model


@dataclass
class ConversionResult:
    output_dir: str
    unet_onnx: str
    vae_onnx: str
    text_encoder_onnx: str
    tokenizer_dir: str
    conversion_time_seconds: float


def export_to_onnx(model_id: str, output_dir: str,
                   opset_version: int = 17, fp32: bool = True) -> ConversionResult:
    if not fp32:
        raise ValueError("La exportacion CPU solo admite FP32.")
    if opset_version < 17:
        raise ValueError("Usa opset_version >= 17 para este exportador.")
    source, _ = resolve_model(model_id)
    output = model_path(output_dir, must_exist=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"La salida debe estar vacia: {output}")
    from optimum.exporters.onnx import main_export

    start = time.perf_counter()
    # Explicit task/library prevent Hub task detection. Never return a partial export as success.
    main_export(str(source), output=output, task="stable-diffusion",
                library_name="diffusers", framework="pt", device="cpu", dtype="fp32",
                opset=opset_version, local_files_only=True, do_validation=True,
                model_kwargs={"safety_checker": None, "feature_extractor": None,
                              "requires_safety_checker": False})
    components = [output / component / "model.onnx"
                  for component in ("unet", "vae_decoder", "text_encoder")]
    for path in components + [output / "model_index.json", output / "tokenizer" / "vocab.json"]:
        if not path.is_file():
            raise RuntimeError(f"Exportacion incompleta: falta {path}")
    return ConversionResult(str(output), *(str(p) for p in components),
                            str(output / "tokenizer"), time.perf_counter() - start)


def quantize_int8_dynamic(model_path: str, output_path: str, model_type: str = "unet") -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = Path(model_path).resolve()
    output = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source or output.exists():
        raise FileExistsError(f"No se sobrescriben modelos: {output}")
    if model_type not in ("unet", "text_encoder"):
        raise ValueError("model_type debe ser unet o text_encoder.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # MatMul only: ConvInteger is not supported by every CPU execution provider.
    quantize_dynamic(str(source), str(output), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], per_channel=True,
                     use_external_data_format=True)
    return str(output)


def quantize_pipeline(onnx_dir: str, quantize_unet: bool = True,
                      quantize_text_encoder: bool = True) -> dict[str, str]:
    results = {}
    for name, enabled in (("unet", quantize_unet), ("text_encoder", quantize_text_encoder)):
        if enabled:
            source = Path(onnx_dir) / name / "model.onnx"
            results[name] = quantize_int8_dynamic(str(source), str(source.with_name("model_int8.onnx")), name)
    return results


def convert_safetensors_to_diffusers(safetensors_path: str, output_dir: str) -> str:
    raise NotImplementedError(
        "Un checkpoint monolitico necesita su YAML original y componentes compatibles. "
        "Esta conversion no esta soportada; prepara un directorio Diffusers completo "
        "dentro de models/. No se descargara un modelo de referencia."
    )
