#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
import yaml

MODEL_ID = "comfyui-z-image-turbo"
BUNDLE_SUBDIRECTORIES = {
    "text_encoders": "split_files/text_encoders",
    "diffusion_models": "split_files/diffusion_models",
    "vae": "split_files/vae",
}
STANDARD_CATEGORIES = (
    "checkpoints", "diffusion_models", "text_encoders", "clip", "clip_vision",
    "vae", "loras", "controlnet", "upscale_models", "embeddings",
)


def resolved_components() -> dict[str, Path]:
    try:
        result = subprocess.run(
            ["synapse-model", "resolve", MODEL_ID, "--json"],
            check=False, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout) if result.stdout else {}
    except (OSError, json.JSONDecodeError):
        return {}
    if result.returncode or not payload.get("ok"):
        return {}
    return {
        str(item["id"]): Path(str(item["path"]))
        for item in payload.get("data", {}).get("components", [])
        if item.get("id") and item.get("path")
    }


def build_config(model_root: Path) -> dict[str, dict[str, str]]:
    paths: dict[str, list[str]] = {
        category: [str(model_root / category)] for category in STANDARD_CATEGORIES
    }
    canonical = model_root / "safetensors/image-generation/comfyui/z-image-turbo"
    for category, subdirectory in BUNDLE_SUBDIRECTORIES.items():
        paths[category].append(str(canonical / subdirectory))
    bundle = resolved_components().get("z-image-bundle")
    if bundle:
        for category, subdirectory in BUNDLE_SUBDIRECTORIES.items():
            paths[category].append(str(bundle / subdirectory))
    normalized = {
        category: list(dict.fromkeys(os.path.normpath(value) for value in values))
        for category, values in paths.items()
    }
    return {"synapse_comfyui": {category: "\n".join(values) for category, values in normalized.items()}}


def write_atomic(path: Path, value: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ComfyUI paths from activated Synapse model bundles")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_atomic(Path(args.output), build_config(Path(args.model_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
