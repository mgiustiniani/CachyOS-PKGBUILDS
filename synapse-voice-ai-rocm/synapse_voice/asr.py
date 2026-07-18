from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import load_config, resolve_model_paths

_LOCK = threading.Lock()


@dataclass(frozen=True)
class Transcription:
    text: str
    backend: str


def _run_xdna(wav: Path, language: str) -> Transcription:
    command = shutil.which("synapse-whisper-xdna")
    if not command:
        raise RuntimeError("synapse-whisper-xdna is not installed")
    result = subprocess.run(
        [command, "transcribe", str(wav), "--language", language, "--json"],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"XDNA transcription failed: {detail}")
    try:
        payload = json.loads(result.stdout)
        text = payload["text"].strip()
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise RuntimeError("XDNA transcription returned invalid JSON") from exc
    return Transcription(text=text, backend="xdna-vitisai")


def _run_whisper_cpp(wav: Path, language: str, use_gpu: bool) -> Transcription:
    _, model = resolve_model_paths()
    with tempfile.TemporaryDirectory(prefix="synapse-whisper-cpp-") as directory:
        output = Path(directory) / "transcript"
        command = [
            "whisper-cli", "-m", str(model), "-f", str(wav),
            "-l", language, "-otxt", "-of", str(output), "-np",
        ]
        if not use_gpu:
            command.append("-ng")
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "whisper.cpp transcription failed")
        transcript = output.with_suffix(".txt")
        text = transcript.read_text().strip() if transcript.exists() else ""
    return Transcription(text=text, backend="whisper.cpp-rocm" if use_gpu else "whisper.cpp-cpu")


def _backend_order(config: dict[str, str], requested: str | None) -> list[str]:
    backend = (requested or config.get("WHISPER_BACKEND", "rocm")).strip().lower()
    if backend not in {"auto", "xdna", "rocm", "cpu"}:
        raise ValueError(f"Unsupported Whisper backend: {backend}")
    if backend in {"auto", "xdna"}:
        default_fallbacks = "rocm,cpu"
        order = ["xdna"]
    elif backend == "rocm":
        default_fallbacks = "cpu"
        order = ["rocm"]
    else:
        default_fallbacks = ""
        order = ["cpu"]
    fallbacks = config.get("WHISPER_FALLBACK", default_fallbacks)
    for value in fallbacks.split(","):
        value = value.strip().lower()
        if value in {"xdna", "rocm", "cpu"} and value not in order:
            order.append(value)
    return order


def transcribe_wav(wav: Path, language: str, backend: str | None = None) -> Transcription:
    config = load_config()
    errors: list[str] = []
    with _LOCK:
        for candidate in _backend_order(config, backend):
            try:
                if candidate == "xdna":
                    return _run_xdna(wav, language)
                return _run_whisper_cpp(wav, language, use_gpu=candidate == "rocm")
            except (FileNotFoundError, RuntimeError) as exc:
                errors.append(f"{candidate}: {exc}")
    raise RuntimeError("All Whisper backends failed: " + "; ".join(errors))
