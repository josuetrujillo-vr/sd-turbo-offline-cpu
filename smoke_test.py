"""Real image generation with every network connection blocked in this process."""

import logging
import os
import sys
import tempfile

from src.runtime import configure_logging, enable_offline

logger = logging.getLogger("smoke_test")


def main() -> int:
    configure_logging()
    enable_offline()
    attempts = []

    def deny_network(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            attempts.append(event)
            raise RuntimeError(f"Acceso de red prohibido en smoke_test: {event}")

    sys.addaudithook(deny_network)
    try:
        # Empty Hub cache proves that models/ contains the required resources.
        with tempfile.TemporaryDirectory(prefix="sd-offline-") as cache:
            os.environ["HF_HOME"] = cache
            os.environ["HF_HUB_CACHE"] = cache
            os.environ["TRANSFORMERS_CACHE"] = cache
            from src.pipeline import (GenerationConfig, configure_threads,
                                      generate_image, load_diffusers_pipeline)
            from src.config import project_path
            from generate import save_image_with_metadata

            configure_threads()
            pipe = load_diffusers_pipeline("sd_turbo", safety_checker=True)
            gen = GenerationConfig("a test photo of a cat", width=256, height=256, seed=0)
            image, metrics = generate_image(pipe, gen)
            if image.size != (256, 256) or attempts:
                raise AssertionError(f"Tamano o red incorrectos: {image.size}, {attempts}")
            output = project_path("output/smoke_test_256x256.png")
            save_image_with_metadata(image, output, gen.prompt, "", metrics["model"], 0, metrics)
            logger.info("SMOKE TEST OK: imagen real, cache vacia y cero intentos de red.")
            return 0
    except Exception:
        logger.exception("SMOKE TEST ERROR; prepara los recursos con python download_model.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
