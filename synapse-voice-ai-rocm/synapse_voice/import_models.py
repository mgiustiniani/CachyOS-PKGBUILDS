from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from .config import archive_candidates, load_config

EXPECTED = {
    "safetensors/voice-ai/huggingface/ResembleAI/chatterbox/ve.pt":
        "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1",
    "safetensors/voice-ai/huggingface/ResembleAI/chatterbox/t3_mtl23ls_v2.safetensors":
        "b1237586127ce98e7800a68e49938eb5092846862aabcb6e17b2fda7889a6c75",
    "safetensors/voice-ai/huggingface/ResembleAI/chatterbox/s3gen.pt":
        "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3",
    "ggml/speech-recognition/whisper.cpp/ggml-large-v3-turbo-q8_0.bin":
        "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def find_source(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else archive_candidates()
    for candidate in candidates:
        if all((candidate / relative).is_file() for relative in EXPECTED):
            return candidate
    raise FileNotFoundError("No complete Synapse Voice AI model archive was found")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or import Synapse Voice AI models")
    parser.add_argument("--source", help="Format-category model root (the directory containing safetensors/ and ggml/)")
    parser.add_argument("--copy", action="store_true", help="Copy models into VOICE_MODEL_ROOT")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    source = find_source(args.source)
    print(f"model source: {source}")
    if not args.skip_hash:
        for relative, expected in EXPECTED.items():
            path = source / relative
            actual = digest(path)
            if actual != expected:
                raise RuntimeError(f"checksum mismatch for {path}: {actual}")
            print(f"verified: {relative}")

    if args.copy:
        config = load_config()
        destination = Path(config.get("VOICE_MODEL_ROOT", "/var/lib/synapse/voice-ai/models"))
        chatterbox_rel = Path(config["CHATTERBOX_MODEL_REL"])
        whisper_rel = Path(config["WHISPER_MODEL_REL"])
        destination.mkdir(parents=True, exist_ok=True)
        target_chatterbox = destination / chatterbox_rel
        target_chatterbox.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / chatterbox_rel, target_chatterbox, dirs_exist_ok=True)
        target_whisper = destination / whisper_rel
        target_whisper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / whisper_rel, target_whisper)
        print(f"models copied to: {destination}")
    else:
        print("external model archive validated; no files copied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
