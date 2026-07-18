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
sudo synapse-model enqueue trellis2 --source web --mode copy --priority 10
synapse-model jobs
synapse-model pause voice-ai
synapse-model resume voice-ai
synapse-model cancel voice-ai
synapse-model retry voice-ai
synapse-model remove-job voice-ai --partials
synapse-model resolve voice-ai
```

Sources accepted by `install` are `auto`, `usb`, `web`, or an explicit archive root. `auto` prefers a complete local archive and falls back to pinned web sources. The manager discovers filesystems labelled `models`, `/mnt/models`, and desktop removable-media mounts.

## Persistent and resumable transfers

Copy and web installations use a deterministic staging directory under the destination filesystem. Failed, paused, cancelled, or interrupted transfers preserve verified partial data. Running the same `install` command again, or using `resume MODEL`, continues that job. Use `--restart` only when partial data should be discarded.

HTTP sources resume through validated Range requests. Hugging Face sources reuse the official `hf download` local cache and Xet transfer implementation. USB copies use `.part` files and continue from the existing byte offset. A disk-space preflight runs before acquisition.

Jobs are stored atomically under `/var/lib/synapse/model-manager/jobs`. `pause` and `cancel` are cooperative: the running transfer observes the request at its next I/O boundary and exits while retaining partial data.

## Background queue

`enqueue` writes a durable request and asks systemd to start `synapse-model-worker.service` when used as root with the default state directory. The worker processes one model at a time, highest priority first, and continues after individual failures. Stale `running` and `interrupted` jobs are resumed after a worker crash or reboot; failed, paused, and cancelled jobs require `retry` or an explicit `resume`.

Enable boot recovery and automatic directory watching once on installed systems:

```sh
sudo systemctl enable --now synapse-model-worker.path synapse-model-worker.service
```

The service runs as root because system model destinations are protected. Its systemd sandbox limits writable locations to `/var/lib/synapse`, `/mnt`, `/media`, and `/run/media`. Queue records never contain Hugging Face tokens. A future desktop frontend should enqueue through a narrow polkit action instead of changing queue-directory permissions.

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

`--jsonl` emits byte progress, rate, ETA, job state, and a final `result` event:

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
