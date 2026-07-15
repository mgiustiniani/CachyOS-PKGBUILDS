from __future__ import annotations

import os
import shlex
from pathlib import Path

SYSTEM_CONFIG = Path("/etc/synapse/voice-ai/models.conf")


def load_config(path: Path = SYSTEM_CONFIG) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                parsed = shlex.split(value, comments=True)
                values[key.strip()] = parsed[0] if parsed else ""
            except ValueError:
                values[key.strip()] = value.strip().strip("\"'")
    for key in tuple(values):
        values[key] = os.environ.get(key, values[key])
    return values


def archive_candidates() -> list[Path]:
    candidates: list[Path] = []
    media_root = Path("/run/media")
    if media_root.exists():
        candidates.extend(media_root.glob("*/models/models"))
    candidates.extend(
        [
            Path("/mnt/models/models"),
            Path("/media/models/models"),
            Path("/run/synapse-models/models"),
        ]
    )
    return candidates


def resolve_model_paths(config: dict[str, str] | None = None) -> tuple[Path, Path]:
    cfg = config or load_config()
    root = Path(cfg.get("VOICE_MODEL_ROOT", "/var/lib/synapse/voice-ai/models"))
    chatterbox_rel = Path(
        cfg.get(
            "CHATTERBOX_MODEL_REL",
            "safetensors/voice-ai/huggingface/ResembleAI/chatterbox",
        )
    )
    whisper_rel = Path(
        cfg.get(
            "WHISPER_MODEL_REL",
            "ggml/speech-recognition/whisper.cpp/ggml-large-v3-turbo-q8_0.bin",
        )
    )

    local_chatterbox = root / chatterbox_rel
    local_whisper = root / whisper_rel
    if local_chatterbox.is_dir() and local_whisper.is_file():
        return local_chatterbox, local_whisper

    configured_source = cfg.get("VOICE_MODEL_SOURCE", "auto")
    candidates = [] if configured_source == "auto" else [Path(configured_source)]
    candidates.extend(archive_candidates())
    for candidate in candidates:
        chatterbox = candidate / chatterbox_rel
        whisper = candidate / whisper_rel
        if chatterbox.is_dir() and whisper.is_file():
            return chatterbox, whisper

    raise FileNotFoundError(
        "Voice AI models were not found. Mount the disk labelled 'models' or run "
        "synapse-voice-import-models."
    )


def voice_directory() -> Path:
    xdg_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    directory = xdg_data / "synapse/voice-ai/voices"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    return directory
