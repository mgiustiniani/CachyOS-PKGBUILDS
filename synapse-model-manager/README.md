# Synapse Model Manager

Shared model lifecycle management for Synapse Linux. Runtime packages and model manifests ship through pacman; large model weights remain outside packages and ISO images.

## Commands

```sh
synapse-model list
synapse-model show voice-ai
synapse-model discover
synapse-model status
synapse-model verify voice-ai --source auto
synapse-model install voice-ai --source usb --mode external
sudo synapse-model install voice-ai --source web --mode copy
synapse-model resolve voice-ai
```

Sources accepted by `install` are `auto`, `usb`, `web`, or an explicit archive root. `auto` prefers a complete local archive and falls back to pinned web sources. The manager discovers filesystems labelled `models`, `/mnt/models`, and desktop removable-media mounts.

Modes:

- `external`: validate and activate the archive in place;
- `copy`: stage, verify, and atomically activate selected files under the destination root.

The canonical archive is format-first:

```text
models/
├── ggml/
├── gguf/
├── pytorch/
└── safetensors/
```

## Machine-readable API

`--json` may appear anywhere on the command line and produces one versioned JSON document on stdout. Progress and diagnostics do not contaminate stdout.

```sh
synapse-model status voice-ai --json
```

`--jsonl` emits progress events followed by a final `result` event:

```sh
sudo synapse-model install trellis2 --source web --mode copy --jsonl
```

Both formats include `schemaVersion`. A failed command emits a structured error and returns a non-zero exit code.

## Manifests

Product packages can install TOML manifests into:

```text
/usr/share/synapse/models.d
```

Additional registries can be selected with `--manifest-dir` or the colon-separated `SYNAPSE_MODEL_MANIFEST_DIRS` environment variable. Manifests pin repositories and revisions, describe required files and globs, and may bind exact sizes and SHA-256 hashes.

The initial registry covers:

- Synapse Voice AI;
- TRELLIS.2;
- SeamlessM4T translation;
- Wan2.1 VACE video generation;
- three DeepSeek V4 Flash GGUF variants usable by DS4 and future llama.cpp integration.

Model licenses remain those of the upstream model authors. The manager license does not alter model licenses.
