#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor

MODEL_ID = "whisper-xdna2"
MODEL_COMPONENT = "whisper-large-v3-turbo-onnx-npu"
PROCESSOR_COMPONENT = "whisper-large-v3-turbo-processor"
CONFIG_ROOT = Path("/usr/share/synapse-whisper-xdna")
LANGUAGES = {
    "de": "german", "en": "english", "es": "spanish", "fr": "french",
    "it": "italian", "ja": "japanese", "ko": "korean", "pl": "polish",
    "pt": "portuguese", "ru": "russian", "zh": "chinese",
}


@contextlib.contextmanager
def native_stdout_to_stderr(enabled: bool):
    """Keep native Vitis/compiler diagnostics out of the JSON stdout API."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
        ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def resolve_components() -> tuple[Path, Path]:
    model_override = os.environ.get("SYNAPSE_WHISPER_XDNA_MODEL")
    processor_override = os.environ.get("SYNAPSE_WHISPER_XDNA_PROCESSOR")
    if model_override and processor_override:
        return Path(model_override), Path(processor_override)
    result = subprocess.run(
        ["synapse-model", "resolve", MODEL_ID, "--json"],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "model resolution failed")
    payload = json.loads(result.stdout)
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("errors") or "model resolution failed"))
    components = {
        item["id"]: Path(item["path"])
        for item in payload["data"]["components"]
    }
    try:
        return components[MODEL_COMPONENT], components[PROCESSOR_COMPONENT]
    except KeyError as exc:
        raise RuntimeError(f"model activation lacks component: {exc.args[0]}") from exc


def validate_paths(model: Path, processor: Path) -> list[str]:
    required = [
        model / "encoder_model.onnx",
        model / "encoder_model.onnx.data",
        model / "decoder_model.onnx",
        processor / "preprocessor_config.json",
        processor / "tokenizer.json",
        processor / "tokenizer_config.json",
        CONFIG_ROOT / "vitisai_config_whisper_encoder.json",
        CONFIG_ROOT / "vitisai_config_whisper_decoder.json",
    ]
    return [str(path) for path in required if not path.is_file()]


def cache_root(model: Path, explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
    else:
        preferred = model / "xdna-cache"
        try:
            preferred.mkdir(parents=True, exist_ok=True)
            test = preferred / ".write-test"
            test.touch()
            test.unlink()
            root = preferred
        except OSError:
            base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            root = base / "synapse/whisper-xdna/large-v3-turbo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def provider_options(config: str, cache: Path, key: str) -> dict[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "config_file": str(CONFIG_ROOT / config),
        "cache_dir": str(cache),
        "cache_key": key,
    }


def create_sessions(model: Path, cache: Path) -> tuple[ort.InferenceSession, ort.InferenceSession, dict[str, float]]:
    lock_path = cache / ".compile.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        encoder = ort.InferenceSession(
            str(model / "encoder_model.onnx"),
            providers=[(
                "VitisAIExecutionProvider",
                provider_options(
                    "vitisai_config_whisper_encoder.json", cache / "encoder",
                    "whisper_large_turbo_encoder",
                ),
            )],
        )
        encoder_load = time.perf_counter() - started
        started = time.perf_counter()
        decoder = ort.InferenceSession(
            str(model / "decoder_model.onnx"),
            providers=[(
                "VitisAIExecutionProvider",
                provider_options(
                    "vitisai_config_whisper_decoder.json", cache / "decoder",
                    "whisper_large_turbo_decoder",
                ),
            )],
        )
        decoder_load = time.perf_counter() - started
    for name, session in (("encoder", encoder), ("decoder", decoder)):
        if "VitisAIExecutionProvider" not in session.get_providers():
            raise RuntimeError(f"VitisAIExecutionProvider failed to initialize for {name}")
    return encoder, decoder, {
        "encoder_session_seconds": encoder_load,
        "decoder_session_seconds": decoder_load,
    }


def convert_audio(source: Path, destination: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(destination),
        ],
        text=True, capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")


def load_chunks(path: Path) -> list[np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2:
            raise RuntimeError("converted audio is not mono 16 kHz PCM16")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)
    audio /= 32768.0
    if not len(audio):
        raise RuntimeError("audio input is empty")
    window = 30 * 16000
    return [audio[offset:offset + window] for offset in range(0, len(audio), window)]


def normalize_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").split("-", 1)[0]
    return LANGUAGES.get(normalized, value)


def decode_chunk(
    chunk: np.ndarray,
    processor: WhisperProcessor,
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    language: str,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    features = processor(
        chunk, sampling_rate=16000, return_tensors="np",
    ).input_features.astype(np.float32)
    started = time.perf_counter()
    encoded = encoder.run(None, {"x": features})[0]
    encoder_seconds = time.perf_counter() - started

    tokenizer = processor.tokenizer
    start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    forced = [token for _, token in processor.get_decoder_prompt_ids(
        language=normalize_language(language), task="transcribe",
    )]
    tokens = [start, *forced]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    suppressed = np.asarray(
        sorted(token for token in set(tokenizer.all_special_ids) if token != eos),
        dtype=np.int64,
    )
    decoder_seconds = 0.0
    for _ in range(max_tokens):
        values = np.full((1, 448), pad, dtype=np.int64)
        values[0, :len(tokens)] = tokens
        started = time.perf_counter()
        logits = decoder.run(None, {"x": values, "xa": encoded})[0]
        decoder_seconds += time.perf_counter() - started
        next_logits = logits[0, len(tokens) - 1].copy()
        next_logits[suppressed] = -np.inf
        token = int(np.argmax(next_logits))
        tokens.append(token)
        if token == eos or len(tokens) >= 448:
            break
    generated = tokens[1 + len(forced):]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {
        "encoder_seconds": encoder_seconds,
        "decoder_seconds": decoder_seconds,
        "tokens": len(generated),
    }


def transcribe(args: argparse.Namespace) -> dict[str, Any]:
    model, processor_root = resolve_components()
    missing = validate_paths(model, processor_root)
    if missing:
        raise RuntimeError("missing required files: " + ", ".join(missing))
    cache = cache_root(model, args.cache_dir)
    processor = WhisperProcessor.from_pretrained(processor_root, local_files_only=True)
    encoder, decoder, load_times = create_sessions(model, cache)
    if "VitisAIExecutionProvider" not in encoder.get_providers() or "VitisAIExecutionProvider" not in decoder.get_providers():
        raise RuntimeError("VitisAIExecutionProvider is not active for both sessions")

    with tempfile.TemporaryDirectory(prefix="synapse-whisper-xdna-") as directory:
        wav = Path(directory) / "input.wav"
        convert_audio(Path(args.input), wav)
        chunks = load_chunks(wav)
        texts: list[str] = []
        timings: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            text, timing = decode_chunk(
                chunk, processor, encoder, decoder, args.language, args.max_tokens,
            )
            texts.append(text)
            timings.append({"index": index, **timing})
    return {
        "schema_version": 1,
        "backend": "xdna-vitisai",
        "model": MODEL_ID,
        "language": args.language,
        "text": " ".join(text for text in texts if text).strip(),
        "chunks": len(chunks),
        "cache": str(cache),
        "providers": {
            "encoder": encoder.get_providers(),
            "decoder": decoder.get_providers(),
        },
        "session": load_times,
        "timings": timings,
    }


def doctor() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "model_manager": shutil.which("synapse-model") is not None,
        "vitisai_provider": "VitisAIExecutionProvider" in ort.get_available_providers(),
    }
    model = processor = None
    try:
        model, processor = resolve_components()
        missing = validate_paths(model, processor)
    except Exception as exc:
        missing = [str(exc)]
    checks["model_complete"] = not missing
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "degraded",
        "checks": checks,
        "model_root": str(model) if model else None,
        "processor_root": str(processor) if processor else None,
        "missing": missing,
    }


def warmup(args: argparse.Namespace) -> dict[str, Any]:
    model, processor = resolve_components()
    missing = validate_paths(model, processor)
    if missing:
        raise RuntimeError("missing required files: " + ", ".join(missing))
    cache = cache_root(model, args.cache_dir)
    encoder, decoder, load_times = create_sessions(model, cache)
    encoded = encoder.run(None, {"x": np.zeros((1, 128, 3000), dtype=np.float32)})[0]
    ids = np.full((1, 448), 50257, dtype=np.int64)
    ids[0, :4] = [50258, 50274, 50360, 50364]
    decoder.run(None, {"x": ids, "xa": encoded})
    return {
        "schema_version": 1,
        "status": "ready",
        "cache": str(cache),
        "providers": {
            "encoder": encoder.get_providers(), "decoder": decoder.get_providers(),
        },
        "session": load_times,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse-whisper-xdna")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "warmup"):
        command = subparsers.add_parser(name)
        command.add_argument("--json", action="store_true")
        if name == "warmup":
            command.add_argument("--cache-dir")
    command = subparsers.add_parser("transcribe")
    command.add_argument("input")
    command.add_argument("--language", default="it")
    command.add_argument("--max-tokens", type=int, default=160)
    command.add_argument("--cache-dir")
    command.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        with native_stdout_to_stderr(args.json):
            if args.command == "doctor":
                result = doctor()
                ok = result["status"] == "ready"
            elif args.command == "warmup":
                result = warmup(args)
                ok = True
            else:
                result = transcribe(args)
                ok = True
        if args.json:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif args.command == "transcribe":
            print(result["text"])
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({
                "schema_version": 1, "status": "error", "error": str(exc),
            }, ensure_ascii=False, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
