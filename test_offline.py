"""Regression checks without model weights. Run: python -m unittest -v test_offline."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from src.runtime import enable_offline, plain_text

enable_offline()


class OfflineTests(unittest.TestCase):
    def test_imports_ui_and_schema_never_attempt_network(self):
        code = """
import sys
attempts = []
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        address = args[0] if event == 'socket.getaddrinfo' else args[-1]
        host = address[0] if isinstance(address, tuple) else address
        if host in ('127.0.0.1', '::1', 'localhost'):
            return  # Windows asyncio uses loopback sockets internally.
        attempts.append(event)
        raise RuntimeError('network forbidden')
sys.addaudithook(deny)
import app, generate, benchmark, src.convert
from fastapi.testclient import TestClient
demo = app.build_ui()
with TestClient(demo.app) as client:
    for route in ('/', '/config', '/info'):
        response = client.get(route)
        assert response.status_code == 200, (route, response.text)
    response = client.post('/api/generate', json={'data': [
        '', '', 'sd_turbo', 'sfw', 1, 0, 256, 256, 0, True, '', 1,
    ]})
    assert response.status_code == 200, response.text
    assert 'ValueError' in response.json()['data'][1], response.text
    response = client.post('/queue/join', json={'data': []})
    assert response.status_code == 400, response.text
assert not demo.analytics_enabled
assert 'fonts.googleapis.com' not in demo.theme._get_theme_css()
demo.close()
assert not attempts, attempts
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_paths_and_missing_models_fail_before_loading(self):
        from src.config import model_path, resolve_model
        with self.assertRaises(ValueError):
            model_path("https://huggingface.co/stabilityai/sd-turbo")
        with self.assertRaises(ValueError):
            model_path("models/../outside")
        with self.assertRaises(FileNotFoundError):
            model_path("models/definitely-missing-offline-test")
        with self.assertRaises(ValueError):
            resolve_model("unknown/model")

    def test_all_model_loaders_receive_local_files_only(self):
        from src.config import load_config
        from src.pipeline import load_diffusers_pipeline
        from diffusers import AutoencoderTiny, StableDiffusionPipeline
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "model_index.json").write_text("{}")
            config = load_config()
            with patch("src.pipeline.resolve_model", return_value=(path, config["models"]["sd_turbo"])), \
                 patch("src.pipeline.model_path", return_value=path), \
                 patch.object(AutoencoderTiny, "from_pretrained") as vae, \
                 patch.object(StableDiffusionPipeline, "from_pretrained") as loader, \
                 patch.object(StableDiffusionSafetyChecker, "from_pretrained") as checker, \
                 patch.object(CLIPImageProcessor, "from_pretrained") as processor:
                load_diffusers_pipeline(safety_checker=True)
                for call in (vae, loader, checker, processor):
                    self.assertIs(call.call_args.kwargs["local_files_only"], True)
                    self.assertEqual(call.call_args.args[0], str(path))
                self.assertIs(loader.call_args.kwargs["vae"], vae.return_value)
                loader.side_effect = OSError("broken weights")
                with self.assertRaisesRegex(OSError, "broken weights"):
                    load_diffusers_pipeline()
                self.assertEqual(loader.call_count, 2)  # No fallback loader.

    def test_seed_zero_metadata_and_nsfw_failure(self):
        from PIL import Image
        from generate import save_image_with_metadata
        from src.pipeline import GenerationConfig, generate_image

        image = Image.new("RGB", (64, 64))
        pipe = MagicMock(return_value=SimpleNamespace(images=[image], nsfw_content_detected=[False]))
        pipe._is_turbo = True
        pipe._model_name = "stabilityai/sd-turbo"
        _, metrics = generate_image(pipe, GenerationConfig("a cat", width=64, height=64, seed=0))
        self.assertEqual(metrics["seed"], 0)
        self.assertEqual(pipe.call_args.kwargs["generator"].initial_seed(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            save_image_with_metadata(image, path, "a cat", "", metrics["model"], 0, metrics)
            with Image.open(path) as result:
                self.assertEqual(result.info["seed"], "0")
                self.assertEqual(json.loads(result.info["metrics"])["steps"], 1)
        pipe.return_value.nsfw_content_detected = [True]
        with self.assertRaisesRegex(ValueError, "NSFW"):
            generate_image(pipe, GenerationConfig("a cat"))

    def test_input_validation_and_guardrails(self):
        from src.pipeline import GenerationConfig, configure_threads
        from src.guardrails import PromptGuardrail
        for kwargs in ({"prompt": ""}, {"width": 513}, {"num_inference_steps": 0},
                       {"seed": -1}, {"seed": 2**63}, {"guidance_scale": float("nan")},
                       {"prompt": "a 5 year old girl"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GenerationConfig(**({"prompt": "a cat"} | kwargs))
        with self.assertRaises(re.error):
            PromptGuardrail(["["])
        configure_threads(2)
        configure_threads(2)

    def test_cache_reuse_profile_switch_and_recovery(self):
        import app
        from src.config import load_config
        app._pipe_cache.update(key=None, pipe=None)
        spec = load_config()["models"]["sd_turbo"]
        with patch("app.resolve_model", return_value=(Path("models/sd-turbo"), spec)), \
             patch("src.pipeline.load_diffusers_pipeline") as loader:
            first = app.get_or_load_pipeline("sd_turbo", "auto", True)
            self.assertIs(first, app.get_or_load_pipeline("sd_turbo", "diffusers_cpu", True))
            self.assertEqual(loader.call_count, 1)
            loader.side_effect = RuntimeError("load failed")
            with self.assertRaises(RuntimeError):
                app.get_or_load_pipeline("sd_turbo", "auto", False)
            self.assertIsNone(app._pipe_cache["pipe"])
            loader.side_effect = None
            app.get_or_load_pipeline("sd_turbo", "auto", False)
            self.assertEqual(loader.call_count, 3)
        app._pipe_cache.update(key=None, pipe=None)

    def test_gradio_errors_are_visible_and_logged(self):
        import app
        with patch("app.get_or_load_pipeline", side_effect=RuntimeError("real failure")), \
             self.assertLogs("app", level="ERROR") as logs:
            image, info, gallery = app.generate("a cat", "", "sd_turbo", "nsfw",
                                               1, 0, 512, 512, 0, True, None, 1)
        self.assertIsNone(image)
        self.assertEqual(gallery, [])
        self.assertIn("RuntimeError: real failure", info)
        self.assertIn("Traceback", "\n".join(logs.output))

    def test_local_adapters_and_scale(self):
        from src.pipeline import apply_lora, load_textual_inversion
        pipe = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.safetensors"
            path.touch()
            with patch("src.pipeline.model_path", return_value=path):
                apply_lora(pipe, str(path), 0.5)
                load_textual_inversion(pipe, str(path))
            self.assertTrue(pipe.load_lora_weights.call_args.kwargs["local_files_only"])
            self.assertTrue(pipe.load_textual_inversion.call_args.kwargs["local_files_only"])
            pipe.set_adapters.assert_called_once_with(["local"], adapter_weights=[0.5])
            pipe.fuse_lora.assert_not_called()

    def test_offline_flags_and_no_emojis_in_owned_code(self):
        os.environ["HF_HUB_OFFLINE"] = "0"
        enable_offline()
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["GRADIO_ANALYTICS_ENABLED"], "False")
        files = list(Path(".").glob("*.py")) + list(Path("src").glob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertEqual(plain_text(text), text, str(path))


if __name__ == "__main__":
    unittest.main()
