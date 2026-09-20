"""Shared configuration and paths; does not enable offline mode on import."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = PROJECT_ROOT / "models"


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config.yaml debe contener un mapa de configuracion.")
    for section in ("models", "vae", "safety_checker", "profiles", "hardware", "generation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Falta la seccion {section} en config.yaml.")
    return config


def model_path(value: str | Path, *, must_exist: bool = True) -> Path:
    """Model data must live under models/, including resolved symbolic links."""
    path = project_path(value)
    if not path.is_relative_to(MODELS_ROOT.resolve()):
        raise ValueError(f"El recurso debe estar dentro de {MODELS_ROOT}: {value}")
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"Falta el recurso local {path}. Ejecuta python download_model.py con internet."
        )
    return path


def resolve_model(value: str | None = None, config: dict | None = None) -> tuple[Path, dict]:
    config = config if config is not None else load_config()
    value = value or config["profiles"]["sfw"]["model"]
    for name, spec in config["models"].items():
        if value in (name, spec["repo"]) or project_path(value) == project_path(spec["path"]):
            return model_path(spec["path"]), spec
    raise ValueError(f"Modelo no configurado: {value}. Usa un alias de config.yaml.")
