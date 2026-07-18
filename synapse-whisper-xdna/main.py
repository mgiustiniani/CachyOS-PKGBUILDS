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
import torch
from transformers import GenerationConfig, WhisperProcessor
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor

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


def load_chunks(
    path: Path, chunk_seconds: float, overlap_seconds: float,
) -> list[tuple[float, np.ndarray]]:
    if not 1.0 <= chunk_seconds <= 30.0:
        raise ValueError("chunk seconds must be between 1 and 30")
    if not 0.0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap seconds must be non-negative and smaller than the chunk")
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getframerate() != 16000 or stream.getsampwidth() != 2:
            raise RuntimeError("converted audio is not mono 16 kHz PCM16")
        audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32)
    audio /= 32768.0
    if not len(audio):
        raise RuntimeError("audio input is empty")
    window = round(chunk_seconds * 16000)
    step = round((chunk_seconds - overlap_seconds) * 16000)
    chunks: list[tuple[float, np.ndarray]] = []
    offset = 0
    while True:
        chunks.append((offset / 16000.0, audio[offset:offset + window]))
        if offset + window >= len(audio):
            break
        candidate = offset + step
        if candidate + window >= len(audio):
            candidate = max(0, len(audio) - window)
        if candidate <= offset:
            break
        offset = candidate
    return chunks


def normalize_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").split("-", 1)[0]
    return LANGUAGES.get(normalized, normalized)


def detect_language(
    encoded: np.ndarray, tokenizer: Any, decoder: ort.InferenceSession,
) -> tuple[str, float]:
    start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    pad = tokenizer.pad_token_id
    values = np.full((1, 448), pad, dtype=np.int64)
    values[0, 0] = start
    started = time.perf_counter()
    logits = decoder.run(None, {"x": values, "xa": encoded})[0][0, 0]
    elapsed = time.perf_counter() - started
    language_ids = [
        token_id for marker, token_id in tokenizer.get_vocab().items()
        if marker.startswith("<|") and marker.endswith("|>")
        and len(marker[2:-2]) in (2, 3) and marker[2:-2].isalpha()
    ]
    if not language_ids:
        raise RuntimeError("processor tokenizer exposes no Whisper language tokens")
    token = language_ids[int(np.argmax(logits[language_ids]))]
    marker = tokenizer.convert_ids_to_tokens(token)
    if not marker.startswith("<|") or not marker.endswith("|>"):
        raise RuntimeError(f"decoder returned invalid language token: {marker}")
    return marker[2:-2], elapsed


def merge_text(previous: str, current: str) -> str:
    if not previous:
        return current.strip()
    if not current:
        return previous.strip()
    left = previous.split()
    right = current.split()
    maximum = min(30, len(left), len(right))
    overlap = 0
    for size in range(maximum, 0, -1):
        if [word.casefold() for word in left[-size:]] == [word.casefold() for word in right[:size]]:
            overlap = size
            break
    return " ".join([*left, *right[overlap:]]).strip()


def append_segments(
    destination: list[dict[str, Any]], incoming: list[dict[str, Any]],
    base_time: float, previous_end: float,
) -> None:
    overlap = max(0.0, previous_end - base_time)
    ownership_boundary = base_time + overlap / 2.0
    if overlap:
        destination[:] = [
            item for item in destination
            if (item["start"] + item["end"]) / 2.0 <= ownership_boundary
        ]
    for segment in incoming:
        absolute = {
            "start": segment["start"] + base_time,
            "end": segment["end"] + base_time,
            "text": segment["text"],
        }
        if overlap and (absolute["start"] + absolute["end"]) / 2.0 < ownership_boundary:
            continue
        duplicate = any(
            item["text"].casefold() == absolute["text"].casefold()
            and abs(item["start"] - absolute["start"]) <= overlap + 0.5
            for item in destination[-3:]
        )
        if not duplicate:
            destination.append(absolute)


def decode_chunk(
    chunk: np.ndarray,
    processor: WhisperProcessor,
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    language: str,
    max_tokens: int,
    timestamps: bool,
) -> tuple[str, list[dict[str, Any]], str, dict[str, Any]]:
    features = processor(
        chunk, sampling_rate=16000, return_tensors="np",
    ).input_features.astype(np.float32)
    started = time.perf_counter()
    encoded = encoder.run(None, {"x": features})[0]
    encoder_seconds = time.perf_counter() - started

    tokenizer = processor.tokenizer
    detected_language = normalize_language(language)
    language_seconds = 0.0
    if detected_language == "auto":
        detected_language, language_seconds = detect_language(encoded, tokenizer, decoder)
    start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    forced = [token for _, token in processor.get_decoder_prompt_ids(
        language=detected_language, task="transcribe",
    )]
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    if timestamps:
        forced = [token for token in forced if token != no_timestamps]
    tokens = [start, *forced]
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    timestamp_begin = no_timestamps + 1
    if timestamps:
        generation = GenerationConfig(
            eos_token_id=eos,
            bos_token_id=tokenizer.bos_token_id,
            no_timestamps_token_id=no_timestamps,
            max_initial_timestamp_index=50,
        )
        timestamp_processor = WhisperTimeStampLogitsProcessor(
            generation, begin_index=len(tokens),
        )
        suppressed = np.asarray(sorted(
            token for token in set(tokenizer.all_special_ids)
            if token != eos and token < timestamp_begin
        ), dtype=np.int64)
    else:
        timestamp_processor = None
        suppressed = np.asarray(sorted(
            token for token in set(tokenizer.all_special_ids) if token != eos
        ), dtype=np.int64)

    decoder_seconds = 0.0
    for _ in range(max_tokens):
        values = np.full((1, 448), pad, dtype=np.int64)
        values[0, :len(tokens)] = tokens
        started = time.perf_counter()
        logits = decoder.run(None, {"x": values, "xa": encoded})[0]
        decoder_seconds += time.perf_counter() - started
        next_logits = logits[0, len(tokens) - 1].copy()
        if timestamp_processor is not None:
            scores = timestamp_processor(
                torch.tensor([tokens]), torch.from_numpy(next_logits[None]),
            )
            scores[0, suppressed] = -float("inf")
            token = int(torch.argmax(scores[0]))
        else:
            next_logits[suppressed] = -np.inf
            token = int(np.argmax(next_logits))
        tokens.append(token)
        if token == eos or len(tokens) >= 448:
            break

    generated = tokens[1 + len(forced):]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    segments: list[dict[str, Any]] = []
    if timestamps:
        decoded = tokenizer.decode(
            generated, decode_with_timestamps=True, output_offsets=True,
        )
        for offset in decoded.get("offsets", []):
            segment_text = offset["text"].strip()
            start_time, end_time = offset["timestamp"]
            if segment_text and end_time is not None:
                segments.append({
                    "start": float(start_time), "end": float(end_time),
                    "text": segment_text,
                })
    return text, segments, detected_language, {
        "encoder_seconds": encoder_seconds,
        "language_seconds": language_seconds,
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
        chunks = load_chunks(wav, args.chunk_seconds, args.overlap_seconds)
        merged_text = ""
        timings: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        selected_language = args.language
        previous_end = 0.0
        for index, (base_time, chunk) in enumerate(chunks):
            text, chunk_segments, detected_language, timing = decode_chunk(
                chunk, processor, encoder, decoder, selected_language, args.max_tokens,
                args.timestamps,
            )
            if selected_language == "auto":
                selected_language = detected_language
            merged_text = merge_text(merged_text, text)
            append_segments(segments, chunk_segments, base_time, previous_end)
            previous_end = base_time + len(chunk) / 16000.0
            timings.append({"index": index, "offset": base_time, **timing})
        if args.timestamps:
            merged_text = " ".join(segment["text"] for segment in segments).strip()
    return {
        "schema_version": 1,
        "backend": "xdna-vitisai",
        "model": MODEL_ID,
        "language": selected_language,
        "text": merged_text,
        "chunks": len(chunks),
        "cache": str(cache),
        "providers": {
            "encoder": encoder.get_providers(),
            "decoder": decoder.get_providers(),
        },
        "session": load_times,
        "timings": timings,
        "segments": segments,
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
    command.add_argument("--language", default="auto")
    command.add_argument("--max-tokens", type=int, default=160)
    command.add_argument("--chunk-seconds", type=float, default=30.0)
    command.add_argument("--overlap-seconds", type=float, default=1.0)
    command.add_argument("--timestamps", action="store_true")
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
