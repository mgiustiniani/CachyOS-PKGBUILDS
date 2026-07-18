#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

CONFIG = Path("/etc/synapse/whisperx/config.conf")


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if CONFIG.is_file():
        for raw in CONFIG.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@contextlib.contextmanager
def native_stdout_to_stderr(enabled: bool):
    """Keep native library diagnostics out of the JSON stdout API."""
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


def runtime_settings(args: argparse.Namespace) -> tuple[Path, str, str | None]:
    config = load_config()
    model_root = Path(args.model_root or config.get("MODEL_ROOT", "/var/lib/synapse/whisperx/models"))
    model_root.mkdir(parents=True, exist_ok=True)
    requested = (args.device or config.get("DEVICE", "rocm")).lower()
    if requested not in {"rocm", "cpu"}:
        raise ValueError(f"unsupported device: {requested}")
    device = "cuda" if requested == "rocm" else "cpu"
    token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or config.get("HUGGINGFACE_TOKEN") or None
    os.environ.setdefault("TORCH_HOME", str(model_root / "torch"))
    os.environ.setdefault("HF_HOME", str(model_root / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(model_root / "huggingface" / "hub"))
    os.environ.setdefault("NLTK_DATA", str(model_root / "nltk"))
    return model_root, device, token


def validate_device(device: str) -> dict[str, Any]:
    import torch

    hip = torch.version.hip
    if device == "cuda" and (not torch.cuda.is_available() or not hip):
        raise RuntimeError("ROCm was requested but PyTorch HIP is unavailable")
    return {
        "requested": "rocm" if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "hip": hip,
        "available": True,
    }


def read_transcript(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("transcript must be a JSON object with a segments array")
    segments = []
    for index, item in enumerate(payload["segments"]):
        if not isinstance(item, dict):
            raise ValueError(f"segment {index} is not an object")
        text = str(item.get("text", "")).strip()
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if text and end > start:
            segments.append({"start": start, "end": end, "text": text})
    if not segments:
        raise ValueError("transcript contains no usable timed segments")
    return {**payload, "segments": segments}


def nltk_data_ready(model_root: Path) -> bool:
    root = model_root / "nltk"
    return (root / "corpora/wordnet.zip").is_file() and (root / "tokenizers/punkt_tab").is_dir()


def download_nltk_data(model_root: Path) -> None:
    root = model_root / "nltk"
    resources = {
        "corpora/wordnet.zip": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
        "tokenizers/punkt_tab.zip": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
    }
    for relative, url in resources.items():
        archive = root / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            temporary = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(archive)
        if relative.startswith("tokenizers/") and not (archive.parent / "punkt_tab").is_dir():
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(archive.parent)


def load_alignment(language: str, device: str, model_root: Path, model_name: str | None):
    if not nltk_data_ready(model_root):
        raise RuntimeError("NLTK alignment data is missing; run synapse-whisperx download first")
    import whisperx

    return whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=model_name,
        model_dir=str(model_root / "alignment"),
    )


def run_alignment(
    audio_path: Path, payload: dict[str, Any], language: str, device: str,
    model_root: Path, model_name: str | None, char_alignments: bool,
) -> dict[str, Any]:
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    model, metadata = load_alignment(language, device, model_root, model_name)
    result = whisperx.align(
        payload["segments"], model, metadata, audio, device,
        return_char_alignments=char_alignments,
    )
    del model
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return {**payload, **result, "language": language, "alignment_backend": "whisperx-rocm" if device == "cuda" else "whisperx-cpu"}


def run_diarization(
    audio_path: Path, payload: dict[str, Any], device: str, model_root: Path,
    model_name: str | None, token: str | None, min_speakers: int | None,
    max_speakers: int | None,
) -> dict[str, Any]:
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    pipeline = DiarizationPipeline(
        model_name=model_name,
        token=token,
        device=torch.device(device),
        cache_dir=str(model_root / "diarization"),
    )
    diarized = pipeline(
        str(audio_path), min_speakers=min_speakers, max_speakers=max_speakers,
    )
    result = whisperx.assign_word_speakers(diarized, payload, fill_nearest=True)
    result["diarization_backend"] = "whisperx-pyannote-rocm" if device == "cuda" else "whisperx-pyannote-cpu"
    result["speakers"] = sorted({
        str(segment["speaker"])
        for segment in result.get("segments", []) if segment.get("speaker")
    })
    return result


def ensure_diarization_acceptance(args: argparse.Namespace) -> None:
    accepted = args.accept_diarization_terms or load_config().get("ACCEPT_DIARIZATION_MODEL_TERMS", "no").lower() == "yes"
    if not accepted:
        raise RuntimeError(
            "diarization model terms were not accepted; pass --accept-diarization-terms "
            "after reviewing the selected pyannote model terms"
        )


def postprocess(args: argparse.Namespace) -> dict[str, Any]:
    model_root, device, token = runtime_settings(args)
    device_info = validate_device(device)
    payload = read_transcript(Path(args.transcript))
    language = args.language or payload.get("language")
    if not language or language == "auto":
        raise ValueError("alignment requires a detected or explicit language")
    if args.diarize:
        ensure_diarization_acceptance(args)
    result = run_alignment(
        Path(args.audio), payload, language, device, model_root,
        args.align_model, args.char_alignments,
    )
    if args.diarize:
        result = run_diarization(
            Path(args.audio), result, device, model_root, args.diarization_model,
            token, args.min_speakers, args.max_speakers,
        )
    result.update({
        "schema_version": 1,
        "status": "ok",
        "postprocessor": "whisperx",
        "device": device_info,
        "model_root": str(model_root),
    })
    return result


def download(args: argparse.Namespace) -> dict[str, Any]:
    model_root, device, token = runtime_settings(args)
    device_info = validate_device(device)
    download_nltk_data(model_root)
    model, _ = load_alignment(args.language, device, model_root, args.align_model)
    del model
    if args.diarization:
        ensure_diarization_acceptance(args)
        from whisperx.diarize import DiarizationPipeline
        import torch
        pipeline = DiarizationPipeline(
            model_name=args.diarization_model, token=token,
            device=torch.device(device), cache_dir=str(model_root / "diarization"),
        )
        del pipeline
    return {
        "schema_version": 1,
        "status": "ready",
        "language": args.language,
        "diarization": args.diarization,
        "device": device_info,
        "model_root": str(model_root),
    }


def doctor(args: argparse.Namespace) -> dict[str, Any]:
    model_root, device, _ = runtime_settings(args)
    checks: dict[str, Any] = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "nltk_data": nltk_data_ready(model_root),
    }
    try:
        import whisperx
        checks["whisperx"] = True
    except Exception as exc:
        checks["whisperx"] = False
        checks["whisperx_error"] = str(exc)
    try:
        checks["device"] = validate_device(device)
    except Exception as exc:
        checks["device"] = {"available": False, "error": str(exc)}
    ready = checks["ffmpeg"] and checks["whisperx"] and checks["nltk_data"] and checks["device"].get("available", False)
    return {
        "schema_version": 1,
        "status": "ready" if ready else "degraded",
        "checks": checks,
        "model_root": str(model_root),
    }


def common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root")
    parser.add_argument("--device", choices=("rocm", "cpu"))
    parser.add_argument("--hf-token")
    parser.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse-whisperx")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("doctor")
    common_options(command)

    command = commands.add_parser("download")
    command.add_argument("--language", required=True)
    command.add_argument("--align-model")
    command.add_argument("--diarization", action="store_true")
    command.add_argument("--diarization-model")
    command.add_argument("--accept-diarization-terms", action="store_true")
    common_options(command)

    command = commands.add_parser("postprocess")
    command.add_argument("audio")
    command.add_argument("transcript")
    command.add_argument("--output", required=True)
    command.add_argument("--language")
    command.add_argument("--align-model")
    command.add_argument("--char-alignments", action="store_true")
    command.add_argument("--diarize", action="store_true")
    command.add_argument("--diarization-model")
    command.add_argument("--min-speakers", type=int)
    command.add_argument("--max-speakers", type=int)
    command.add_argument("--accept-diarization-terms", action="store_true")
    common_options(command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        with native_stdout_to_stderr(args.json):
            if args.command == "doctor":
                result = doctor(args)
                ok = result["status"] == "ready"
            elif args.command == "download":
                result = download(args)
                ok = True
            else:
                result = postprocess(args)
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                ok = True
        if args.json:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif args.command == "postprocess":
            print(args.output)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if ok else 1
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"schema_version": 1, "status": "error", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
