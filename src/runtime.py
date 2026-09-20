"""Offline defaults and console formatting, before third-party imports."""

import logging
import os
import re
import sys


def enable_offline() -> None:
    # Assignment is intentional: inherited online settings must not win.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[name] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2300-\u23FF"
    "\u200D\uFE0F\u20E3\u2B50\u2B55]"
)


def plain_text(value: str) -> str:
    return _EMOJI.sub("", value)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return plain_text(super().format(record))


def configure_logging(verbose: bool = False) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    handler = logging.StreamHandler()
    handler.setFormatter(PlainFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    ))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=[handler], force=True)
