from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import wave
from array import array
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("/etc/synapse/dub/config.conf")
BACKEND_ROOT = Path("/opt/synapse-dub/backends")
LANGUAGE_NAMES = {
    "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "ru": "Russian", "zh": "Chinese",
}


def language_description(code: str) -> str:
    normalized = code.strip().lower().replace("_", "-")
    base = normalized.split("-", 1)[0]
    name = LANGUAGE_NAMES.get(base)
    return f"{name} (language code: {code})" if name else f"language code {code}"


def load_config(path: Path = CONFIG_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for key in tuple(values):
        values[key] = os.environ.get(key, values[key])
        if key.endswith(("_PATH", "_ROOT", "_CHECKPOINT", "_UNET", "_CONFIG", "_S3FD")):
            values[key] = str(Path(os.path.expandvars(values[key])).expanduser())
    return values


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex_quote(item) for item in command), file=sys.stderr)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture,
        )
    except subprocess.CalledProcessError as exc:
        if capture and exc.stderr:
            raise RuntimeError(f"{command[0]} failed: {exc.stderr.strip()}") from exc
        raise


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def require_command(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"Required command not found: {name}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalize_script(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        data = {"segments": data}
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("Dialogue script must contain a segments array")
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(data["segments"]):
        if not isinstance(item, dict):
            raise ValueError(f"Segment {index} is not an object")
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end <= start:
            raise ValueError(f"Segment {index} has invalid timing")
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "id": item.get("id", index),
                "start": start,
                "end": end,
                "speaker": str(item.get("speaker", "SPEAKER_00")),
                "text": text,
            }
        )
    if not segments:
        raise ValueError("Dialogue script contains no usable segments")
    normalized = {"language": data.get("language"), "segments": segments}
    if data.get("transcription_backend"):
        normalized["transcription_backend"] = data["transcription_backend"]
    return normalized


def media_duration(path: Path) -> float:
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise RuntimeError(f"Media has no positive duration: {path}")
    return duration


def extract_audio(video: Path, output: Path) -> None:
    require_command("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def download_youtube(
    url: str, workspace: Path, force: bool = False, cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("YouTube input must be an HTTP or HTTPS URL")
    require_command("yt-dlp")
    workspace.mkdir(parents=True, exist_ok=True)
    marker = workspace / "youtube-source.path"
    if marker.exists() and not force:
        cached = Path(marker.read_text().strip())
        if cached.is_file():
            return cached
    if force:
        for old in workspace.glob("youtube-source.*"):
            if old != marker and old.is_file():
                old.unlink()
    command = [
        "yt-dlp",
        "--no-playlist",
        "--write-info-json",
        "--merge-output-format",
        "mp4",
        "--print",
        "after_move:filepath",
        "-f",
        "bv*+ba/b",
        "-o",
        str(workspace / "youtube-source.%(ext)s"),
    ]
    if cookies:
        command.extend(("--cookies", str(cookies.expanduser().resolve())))
    if cookies_from_browser:
        command.extend(("--cookies-from-browser", cookies_from_browser))
    command.append(url)
    result = run(command, capture=True)
    paths = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    source = next((path for path in reversed(paths) if path.is_file()), None)
    if source is None:
        raise RuntimeError("yt-dlp completed without producing a source video")
    source = source.resolve()
    marker.write_text(str(source) + "\n")
    return source


def transcribe_xdna(audio: Path, output: Path, language: str | None) -> dict[str, Any]:
    if not language:
        raise RuntimeError("XDNA transcription requires --source-language")
    require_command("synapse-whisper-xdna")
    result = run([
        "synapse-whisper-xdna", "transcribe", str(audio),
        "--language", language, "--timestamps", "--json",
    ], capture=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("synapse-whisper-xdna returned invalid JSON") from exc
    data = {
        "language": language,
        "transcription_backend": payload.get("backend", "xdna-vitisai"),
        "segments": [
            {
                "id": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": "SPEAKER_00",
                "text": str(segment["text"]).strip(),
            }
            for index, segment in enumerate(payload.get("segments", []))
            if str(segment.get("text", "")).strip()
        ],
    }
    data = normalize_script(data)
    write_json(output, data)
    return data


def transcribe_openai(audio: Path, output: Path, model_name: str, language: str | None) -> dict[str, Any]:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("python-openai-whisper is required") from exc
    print(f"Loading Whisper model {model_name}...", file=sys.stderr)
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio), language=language, word_timestamps=True)
    data = {
        "language": result.get("language", language),
        "transcription_backend": "openai-whisper-rocm",
        "segments": [
            {
                "id": segment.get("id", index),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker": "SPEAKER_00",
                "text": segment["text"].strip(),
            }
            for index, segment in enumerate(result.get("segments", []))
            if segment.get("text", "").strip()
        ],
    }
    data = normalize_script(data)
    write_json(output, data)
    return data


def transcribe(
    audio: Path, output: Path, model_name: str, language: str | None, backend: str,
) -> dict[str, Any]:
    if backend in {"xdna", "auto"}:
        try:
            return transcribe_xdna(audio, output, language)
        except RuntimeError as exc:
            print(f"XDNA ASR unavailable, falling back to ROCm: {exc}", file=sys.stderr)
    return transcribe_openai(audio, output, model_name, language)


def diarize(audio: Path, script: dict[str, Any], token: str | None, model: str) -> dict[str, Any]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not available. Run synapse-dub-runtime setup or provide a speaker-labelled script."
        ) from exc
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN is required for pyannote diarization")
    pipeline = Pipeline.from_pretrained(model, token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    result = pipeline(str(audio))
    annotation = getattr(result, "speaker_diarization", result)
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    for segment in script["segments"]:
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for start, end, speaker in turns:
            overlap = max(0.0, min(segment["end"], end) - max(segment["start"], start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        segment["speaker"] = best_speaker
    return script


def speaker_ids(script: dict[str, Any]) -> list[str]:
    return sorted({str(segment["speaker"]) for segment in script["segments"]})


def extract_candidate_samples(
    audio: Path, script: dict[str, Any], directory: Path, seconds_per_speaker: float
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for speaker in speaker_ids(script):
        segments = [
            segment
            for segment in script["segments"]
            if segment["speaker"] == speaker and segment["end"] - segment["start"] >= 1.0
        ]
        segments.sort(key=lambda item: item["end"] - item["start"], reverse=True)
        selected: list[dict[str, Any]] = []
        duration = 0.0
        for segment in segments:
            if duration >= seconds_per_speaker:
                break
            selected.append(segment)
            duration += segment["end"] - segment["start"]
        if not selected:
            continue
        filters: list[str] = []
        labels: list[str] = []
        for index, segment in enumerate(selected):
            filters.append(
                f"[0:a]atrim=start={segment['start']:.3f}:end={segment['end']:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
            labels.append(f"[a{index}]")
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]")
        output = directory / f"candidate-{speaker}.wav"
        run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio),
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-t",
                str(seconds_per_speaker),
                "-ar",
                "24000",
                "-ac",
                "1",
                str(output),
            ]
        )
        outputs.append(output)
    return outputs


def collect_samples(paths: list[Path]) -> list[Path]:
    samples: list[Path] = []
    for path in paths:
        if path.is_dir():
            samples.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}))
        elif path.is_file():
            samples.append(path)
    unique: list[Path] = []
    seen: set[Path] = set()
    for sample in samples:
        resolved = sample.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def preview_sample(path: Path) -> None:
    player = shutil.which("ffplay")
    if not player:
        print("ffplay is not installed; preview unavailable.", file=sys.stderr)
        return
    subprocess.run(
        [player, "-nodisp", "-autoexit", "-loglevel", "error", str(path)],
        check=False,
    )


def map_speakers_interactively(
    script: dict[str, Any], samples: list[Path], output: Path
) -> dict[str, str]:
    if not sys.stdin.isatty():
        raise RuntimeError("Interactive speaker mapping requires a terminal; use --speaker-map FILE")
    if not samples:
        raise RuntimeError("No voice samples are available")
    speakers = speaker_ids(script)
    print("\nDetected dialogue speakers:")
    for speaker in speakers:
        examples = [segment["text"] for segment in script["segments"] if segment["speaker"] == speaker][:3]
        print(f"  {speaker}: {' | '.join(examples)}")
    print("\nAvailable voice samples:")
    for index, sample in enumerate(samples, 1):
        print(f"  {index}) {sample}")
    print("Enter a sample number, or 'p N' to preview sample N. Samples may be reused.")
    mapping: dict[str, str] = {}
    for speaker in speakers:
        while True:
            answer = input(f"Sample for {speaker}: ").strip()
            if answer.lower().startswith("p "):
                try:
                    preview_sample(samples[int(answer.split()[1]) - 1])
                except (ValueError, IndexError):
                    print("Invalid preview selection.")
                continue
            try:
                selected = samples[int(answer) - 1]
            except (ValueError, IndexError):
                print(f"Choose a number from 1 to {len(samples)}.")
                continue
            mapping[speaker] = str(selected)
            break
    print("\nSpeaker mapping:")
    for speaker, sample in mapping.items():
        print(f"  {speaker} -> {sample}")
    confirmation = input("Confirm this mapping? [y/N]: ").strip().lower()
    if confirmation not in {"y", "yes"}:
        raise RuntimeError("Speaker mapping was not confirmed")
    write_json(output, mapping)
    return mapping


def translate_script(
    script: dict[str, Any], target_language: str, endpoint: str, model: str, api_key: str | None
) -> dict[str, Any]:
    source = [
        {
            "text": segment["text"],
            "duration_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
            "max_characters": max(8, round((float(segment["end"]) - float(segment["start"])) * 13)),
        }
        for segment in script["segments"]
    ]
    language = language_description(target_language)
    prompt = (
        f"Translate each timed spoken line to {language}. Return only a JSON array with exactly "
        f"{len(source)} strings in the same order. Each translation must sound natural when spoken, "
        "must fit comfortably inside duration_seconds, and must not exceed max_characters. "
        "Compress wording while preserving the essential meaning and names. Input: "
        + json.dumps(source, ensure_ascii=False)
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise audiovisual dialogue translator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    translated = json.loads(content)
    if not isinstance(translated, list) or len(translated) != len(source):
        raise RuntimeError("Translation endpoint returned an invalid segment list")
    output = json.loads(json.dumps(script))
    output["language"] = target_language
    for segment, text in zip(output["segments"], translated, strict=True):
        segment["text"] = str(text).strip()
    return output


def translate_script_specialist(
    script: dict[str, Any], target_language: str, source_language: str, endpoint: str
) -> dict[str, Any]:
    texts = [str(segment["text"]) for segment in script["segments"]]
    payload = json.dumps(
        {
            "texts": texts,
            "source_language": source_language,
            "target_language": target_language,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/text/translations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    translated = body.get("translations")
    if not isinstance(translated, list) or len(translated) != len(texts):
        raise RuntimeError("Specialist translation server returned an invalid segment list")
    output = json.loads(json.dumps(script))
    output["language"] = target_language
    output["translation_backend"] = "seamless-m4t-v2-large"
    for segment, text in zip(output["segments"], translated, strict=True):
        segment["text"] = str(text).strip()
    return output


def customize_script(
    script: dict[str, Any], brief: str, target_language: str, endpoint: str, model: str,
    api_key: str | None,
) -> dict[str, Any]:
    source = [
        {
            "speaker": segment["speaker"],
            "duration_seconds": round(float(segment["end"]) - float(segment["start"]), 3),
            "text": segment["text"],
        }
        for segment in script["segments"]
    ]
    language = language_description(target_language)
    prompt = (
        f"Rewrite the dialogue as a completely new audiovisual scene in {language}. "
        "Follow the creative brief, preserve the exact segment count and speaker order, and keep "
        "every rewritten line short enough for its duration. Return only a JSON array of strings. "
        f"Creative brief: {brief}\nSource timing and dialogue: "
        + json.dumps(source, ensure_ascii=False)
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You rewrite timed dialogue for lip-synchronized video production.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    content = body["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    rewritten = json.loads(content)
    if not isinstance(rewritten, list) or len(rewritten) != len(script["segments"]):
        raise RuntimeError("Custom dialogue endpoint returned an invalid segment list")
    output = json.loads(json.dumps(script))
    output["language"] = target_language
    output["custom_brief"] = brief
    for segment, text in zip(output["segments"], rewritten, strict=True):
        segment["source_text"] = segment["text"]
        segment["text"] = str(text).strip()
    return output


def coalesce_synthesis_segments(script: dict[str, Any], max_gap: float = 0.25) -> dict[str, Any]:
    output = json.loads(json.dumps(script))
    merged: list[dict[str, Any]] = []
    for source in output["segments"]:
        segment = dict(source)
        if (
            merged
            and merged[-1]["speaker"] == segment["speaker"]
            and float(segment["start"]) - float(merged[-1]["end"]) <= max_gap
        ):
            previous = merged[-1]
            previous["text"] = (str(previous["text"]).rstrip() + " " + str(segment["text"]).lstrip()).strip()
            previous["end"] = segment["end"]
            if "source_text" in previous or "source_text" in segment:
                previous["source_text"] = (
                    str(previous.get("source_text", "")).rstrip()
                    + " " + str(segment.get("source_text", "")).lstrip()
                ).strip()
        else:
            merged.append(segment)
    output["segments"] = merged
    return output


def xtts_synthesize(
    script: dict[str, Any], mapping: dict[str, str], directory: Path, language: str
) -> list[tuple[dict[str, Any], Path]]:
    config = load_config()
    accepted = config.get("ACCEPT_NONCOMMERCIAL_MODEL_LICENSES", "no").lower() in {"yes", "true", "1"}
    if not accepted:
        raise RuntimeError(
            "XTTS-v2 uses the non-commercial Coqui Public Model License. Set "
            "ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=yes after reviewing the license."
        )
    try:
        import torch
        from TTS.api import TTS
    except ImportError as exc:
        raise RuntimeError("Coqui TTS runtime is unavailable; reinstall synapse-dub") from exc
    directory.mkdir(parents=True, exist_ok=True)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    if torch.cuda.is_available():
        tts = tts.to("cuda")
    generated: list[tuple[dict[str, Any], Path]] = []
    for index, segment in enumerate(script["segments"]):
        speaker = segment["speaker"]
        if speaker not in mapping:
            raise RuntimeError(f"No confirmed voice sample for {speaker}")
        output = directory / f"segment-{index:05d}.wav"
        tts.tts_to_file(
            text=segment["text"],
            speaker_wav=mapping[speaker],
            language=language,
            file_path=str(output),
        )
        generated.append((segment, output))
    return generated


def normalize_clip(source: Path, destination: Path, sample_rate: int = 24000) -> None:
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter:a",
            "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:stop_periods=-1:stop_duration=0.10:stop_threshold=-45dB",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )


def read_pcm16(path: Path) -> array:
    with wave.open(str(path), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
            raise RuntimeError(f"Unexpected normalized WAV format: {path}")
        values = array("h")
        values.frombytes(stream.readframes(stream.getnframes()))
        return values


def atempo_chain(speed: float) -> str:
    factors: list[float] = []
    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0
    factors.append(speed)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def assemble_timeline(
    generated: list[tuple[dict[str, Any], Path]], output: Path, minimum_duration: float = 0.0
) -> None:
    sample_rate = 24000
    normalized_dir = output.parent / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[int, array]] = []
    final_samples = round(minimum_duration * sample_rate)
    for index, (segment, source) in enumerate(generated):
        normalized = normalized_dir / f"{index:05d}.wav"
        normalize_clip(source, normalized, sample_rate)
        values = read_pcm16(normalized)
        start = float(segment["start"])
        end = float(segment["end"])
        slot_samples = max(1, round((end - start) * sample_rate))
        if len(values) > slot_samples:
            speed = len(values) / slot_samples
            fitted = normalized_dir / f"{index:05d}-fitted.wav"
            print(
                f"Fitting segment {index}: {len(values) / sample_rate:.3f}s -> "
                f"{slot_samples / sample_rate:.3f}s ({speed:.3f}x)", file=sys.stderr,
            )
            run(
                [
                    "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(normalized),
                    "-filter:a", f"{atempo_chain(speed)},atrim=duration={slot_samples / sample_rate:.8f}",
                    "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(fitted),
                ]
            )
            values = read_pcm16(fitted)
        values = values[:slot_samples]
        offset = round(start * sample_rate)
        clips.append((offset, values))
        final_samples = max(final_samples, round(end * sample_rate))
    mix = array("h", [0]) * final_samples
    for offset, values in clips:
        for index, value in enumerate(values):
            position = offset + index
            mixed = mix[position] + value
            mix[position] = max(-32768, min(32767, mixed))
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(mix.tobytes())


def lipsync(backend: str, video: Path, audio: Path, output: Path, config: dict[str, str]) -> None:
    if backend == "none":
        shutil.copyfile(video, output)
        return
    python = sys.executable
    if backend != "wav2lip":
        raise ValueError(f"Unsupported lip-sync backend in this release: {backend}")
    root = BACKEND_ROOT / "wav2lip"
    checkpoint = Path(config["WAV2LIP_CHECKPOINT"])
    os.environ["WAV2LIP_S3FD"] = config["WAV2LIP_S3FD"]
    (output.parent / "temp").mkdir(parents=True, exist_ok=True)
    run(
        [python, str(root / "inference.py"), "--checkpoint_path", str(checkpoint),
         "--face", str(video), "--audio", str(audio), "--outfile", str(output)],
        cwd=output.parent,
    )


def mux(video: Path, audio: Path, output: Path) -> None:
    run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output),
        ]
    )


def doctor() -> int:
    config = load_config()
    checks: list[tuple[str, bool, str]] = []
    for command in ("ffmpeg", "ffprobe", "python"):
        checks.append((command, shutil.which(command) is not None, shutil.which(command) or "not found"))
    imports = {
        "whisper": "python-openai-whisper",
        "torch": "python-pytorch-opt-rocm",
        "cv2": "python-opencv",
        "TTS": "packaged Coqui TTS runtime",
        "pyannote.audio": "packaged pyannote.audio runtime",
    }
    for module, package in imports.items():
        result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True)
        checks.append((module, result.returncode == 0, package))
    for backend in ("wav2lip",):
        root = BACKEND_ROOT / backend
        checks.append((backend, root.is_dir(), str(root)))
    model_paths = {
        "wav2lip model": config.get("WAV2LIP_CHECKPOINT", ""),
        "wav2lip face model": config.get("WAV2LIP_S3FD", ""),
    }
    for name, path in model_paths.items():
        checks.append((name, bool(path) and Path(path).exists(), path or "not configured"))
    failed = False
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name:22} {detail}")
        failed |= not ok
    return 1 if failed else 0


def prepare(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, str]]:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    audio = workspace / "source-16k.wav"
    if not audio.exists() or args.force:
        extract_audio(args.video.resolve(), audio)
    script_path = workspace / "dialogue.json"
    if args.script:
        script = normalize_script(read_json(args.script))
        write_json(script_path, script)
    elif script_path.exists() and not args.force:
        script = normalize_script(read_json(script_path))
    else:
        asr_backend = args.asr_backend or load_config().get("ASR_BACKEND", "openai")
        script = transcribe(
            audio, script_path, args.whisper_model, args.source_language, asr_backend,
        )
    if args.diarize:
        script = diarize(
            audio,
            script,
            os.environ.get("HUGGINGFACE_TOKEN") or load_config().get("HUGGINGFACE_TOKEN"),
            args.diarization_model,
        )
        write_json(script_path, script)
    candidate_dir = workspace / "samples" / "detected"
    detected = extract_candidate_samples(audio, script, candidate_dir, args.sample_seconds)
    provided = collect_samples([Path(item) for item in args.samples])
    all_samples = collect_samples([*provided, *detected])
    map_path = workspace / "speaker-map.json"
    if args.speaker_map:
        mapping = {str(key): str(Path(value).resolve()) for key, value in read_json(args.speaker_map).items()}
        write_json(map_path, mapping)
    elif map_path.exists() and not args.remap:
        mapping = read_json(map_path)
    else:
        mapping = map_speakers_interactively(script, all_samples, map_path)
    mapping = {str(key): str(Path(value).expanduser().resolve()) for key, value in mapping.items()}
    missing = sorted(set(speaker_ids(script)) - set(mapping))
    if missing:
        raise RuntimeError(f"Speaker map is incomplete; missing: {', '.join(missing)}")
    missing_files = sorted(path for path in mapping.values() if not Path(path).is_file())
    if missing_files:
        raise RuntimeError("Speaker map references missing samples: " + ", ".join(missing_files))
    write_json(map_path, mapping)
    return workspace, script, mapping


def command_prepare(args: argparse.Namespace) -> int:
    workspace, script, mapping = prepare(args)
    print(f"Workspace: {workspace}")
    print(f"Segments: {len(script['segments'])}; speakers: {len(speaker_ids(script))}")
    print(f"Confirmed mappings: {len(mapping)}")
    return 0


def complete_dub(
    args: argparse.Namespace, workspace: Path, script: dict[str, Any], mapping: dict[str, str]
) -> int:
    config = load_config()
    endpoint = args.translation_endpoint or config.get("TRANSLATION_ENDPOINT", "http://127.0.0.1:8000/v1")
    model = args.translation_model or config.get("TRANSLATION_MODEL", "deepseek-v4-flash")
    translation_backend = getattr(args, "translation_backend", None) or config.get("TRANSLATION_BACKEND", "llm")
    specialist_endpoint = (
        getattr(args, "specialist_translation_endpoint", None)
        or config.get("SPECIALIST_TRANSLATION_ENDPOINT", "http://127.0.0.1:8091/v1")
    )
    if getattr(args, "custom_prompt", None):
        script = customize_script(
            script,
            args.custom_prompt,
            args.target_language,
            endpoint,
            model,
            os.environ.get("OPENAI_API_KEY"),
        )
        write_json(workspace / "dialogue-custom.json", script)
    elif args.target_language:
        if translation_backend == "specialist":
            source_language = args.source_language or script.get("language") or "en"
            script = translate_script_specialist(
                script, args.target_language, source_language, specialist_endpoint
            )
        else:
            script = translate_script(
                script,
                args.target_language,
                endpoint,
                model,
                os.environ.get("OPENAI_API_KEY"),
            )
        write_json(workspace / "dialogue-translated.json", script)
    language = args.target_language or script.get("language") or args.source_language
    if not language:
        raise RuntimeError("XTTS language is unknown; set --target-language or --source-language")
    synthesis_script = coalesce_synthesis_segments(script)
    if len(synthesis_script["segments"]) != len(script["segments"]):
        print(
            f"Coalesced {len(script['segments'])} transcript segments into "
            f"{len(synthesis_script['segments'])} speaker utterances", file=sys.stderr,
        )
    generated = xtts_synthesize(synthesis_script, mapping, workspace / "tts-segments", language)
    dubbed_audio = workspace / "dubbed.wav"
    assemble_timeline(generated, dubbed_audio, media_duration(args.video.resolve()))
    synced_video = workspace / f"lipsync-{args.backend}.mp4"
    lipsync(args.backend, args.video.resolve(), dubbed_audio, synced_video, config)
    mux(synced_video, dubbed_audio, args.output.resolve())
    print(args.output.resolve())
    return 0


def command_run(args: argparse.Namespace) -> int:
    workspace, script, mapping = prepare(args)
    return complete_dub(args, workspace, script, mapping)


def command_generative(args: argparse.Namespace) -> int:
    require_command("synapse-video-gen")
    workspace, script, mapping = prepare(args)
    generated_video = workspace / "generative-visual.mp4"
    command = [
        "synapse-video-gen", "generate", "--input", str(args.video.resolve()),
        "--prompt", args.visual_prompt, "--output", str(generated_video),
        "--width", str(args.visual_width), "--height", str(args.visual_height),
        "--fps", str(args.visual_fps), "--steps", str(args.visual_steps),
        "--seed", str(args.visual_seed), "--max-duration", str(args.visual_max_duration),
    ]
    if args.force or not generated_video.is_file():
        run(command)
    args.video = generated_video
    args.custom_prompt = args.dialogue_prompt
    return complete_dub(args, workspace, script, mapping)


def command_youtube(args: argparse.Namespace) -> int:
    args.workspace = args.workspace.resolve()
    args.video = download_youtube(
        args.url, args.workspace, args.force, args.cookies, args.cookies_from_browser
    )
    return command_run(args)


def add_prepare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=Path("synapse-dub-work"))
    parser.add_argument("--script", type=Path, help="Pre-labelled dialogue JSON")
    parser.add_argument("--samples", action="append", default=[], help="Sample file or directory; repeatable")
    parser.add_argument("--speaker-map", type=Path, help="Pre-confirmed speaker-to-sample JSON for non-interactive runs")
    parser.add_argument("--source-language")
    parser.add_argument("--whisper-model", default="large-v3-turbo")
    parser.add_argument(
        "--asr-backend", choices=("openai", "xdna", "auto"),
        help="Speech recognition backend; defaults to ASR_BACKEND from config",
    )
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--sample-seconds", type=float, default=12.0)
    parser.add_argument("--remap", action="store_true")
    parser.add_argument("--force", action="store_true")


def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path)
    add_prepare_options(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse-dub", description="ROCm video dubbing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser("doctor", help="Check runtime and model availability")
    doctor_parser.set_defaults(handler=lambda args: doctor())
    prepare_parser = subparsers.add_parser("prepare", help="Transcribe, diarize, extract samples, and map speakers")
    add_prepare_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=command_prepare)
    run_parser = subparsers.add_parser("run", help="Run the complete dubbing pipeline")
    add_prepare_arguments(run_parser)
    run_parser.add_argument("--target-language", help="Enable optional translation and set XTTS output language")
    run_parser.add_argument("--translation-backend", choices=("llm", "specialist"))
    run_parser.add_argument("--translation-endpoint")
    run_parser.add_argument("--translation-model")
    run_parser.add_argument("--specialist-translation-endpoint")
    run_parser.add_argument("--backend", choices=("wav2lip", "none"), default="wav2lip")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=command_run, custom_prompt=None)

    custom_parser = subparsers.add_parser(
        "custom", help="Create a new scripted scene from a real source video"
    )
    add_prepare_arguments(custom_parser)
    custom_parser.add_argument("--prompt", dest="custom_prompt", required=True, help="Creative dialogue brief")
    custom_parser.add_argument("--target-language", required=True, help="Language of the new dialogue")
    custom_parser.add_argument("--translation-endpoint")
    custom_parser.add_argument("--translation-model")
    custom_parser.add_argument("--backend", choices=("wav2lip", "none"), default="wav2lip")
    custom_parser.add_argument("--output", type=Path, required=True)
    custom_parser.set_defaults(handler=command_run)

    youtube_parser = subparsers.add_parser(
        "youtube", help="Download an authorized YouTube video and translate its dialogue"
    )
    youtube_parser.add_argument("url")
    add_prepare_options(youtube_parser)
    youtube_parser.add_argument("--cookies", type=Path, help="Netscape cookies file for authorized access")
    youtube_parser.add_argument("--cookies-from-browser", help="Browser profile understood by yt-dlp")
    youtube_parser.add_argument("--target-language", required=True)
    youtube_parser.add_argument("--translation-backend", choices=("llm", "specialist"))
    youtube_parser.add_argument("--translation-endpoint")
    youtube_parser.add_argument("--translation-model")
    youtube_parser.add_argument("--specialist-translation-endpoint")
    youtube_parser.add_argument("--backend", choices=("none", "wav2lip"), default="none")
    youtube_parser.add_argument("--output", type=Path, required=True)
    youtube_parser.set_defaults(handler=command_youtube, custom_prompt=None)

    generative_parser = subparsers.add_parser(
        "generative", help="Generate new visual scenes, dialogue, cloned voices, and final video"
    )
    add_prepare_arguments(generative_parser)
    generative_parser.add_argument("--visual-prompt", required=True)
    generative_parser.add_argument("--dialogue-prompt", help="Optional creative brief replacing source dialogue")
    generative_parser.add_argument("--target-language", required=True)
    generative_parser.add_argument("--translation-backend", choices=("llm", "specialist"))
    generative_parser.add_argument("--translation-endpoint")
    generative_parser.add_argument("--translation-model")
    generative_parser.add_argument("--specialist-translation-endpoint")
    generative_parser.add_argument("--backend", choices=("wav2lip", "none"), default="wav2lip")
    generative_parser.add_argument("--visual-width", type=int, default=832)
    generative_parser.add_argument("--visual-height", type=int, default=480)
    generative_parser.add_argument("--visual-fps", type=int, default=16)
    generative_parser.add_argument("--visual-steps", type=int, default=30)
    generative_parser.add_argument("--visual-seed", type=int, default=42)
    generative_parser.add_argument("--visual-max-duration", type=float, default=60.0)
    generative_parser.add_argument("--output", type=Path, required=True)
    generative_parser.set_defaults(handler=command_generative, custom_prompt=None)
    return parser


def main() -> int:
    try:
        parser = build_parser()
        args = parser.parse_args()
        return int(args.handler(args))
    except (RuntimeError, ValueError, subprocess.CalledProcessError, OSError) as exc:
        print(f"synapse-dub: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
