# Synapse Dub

ROCm-first video dubbing orchestration for Synapse Linux:

1. audio extraction with system FFmpeg;
2. transcription with the official Arch `python-openai-whisper` package;
3. optional pyannote speaker diarization or a pre-labelled dialogue script;
4. optional translation through an OpenAI-compatible endpoint;
5. mandatory, explicit speaker-to-voice-sample confirmation;
6. XTTS-v2 multilingual voice cloning;
7. Wav2Lip lip-sync adapter for the first release;
8. final FFmpeg mux.

## Video-to-voice-samples mode

`synapse-dub prepare` can take only a source video. After transcription and
optional diarization it extracts clean candidate speech intervals for every
detected speaker into `WORKSPACE/samples/detected/`. These candidates are not
silently assigned. The user still sees all dialogue speakers and all candidate
or supplied sample files, previews them if desired, and confirms each mapping.
The confirmed mapping is stored as JSON for reproducible later runs.

```sh
export HUGGINGFACE_TOKEN=...
synapse-dub prepare source.mp4 \
  --workspace dubbing-work \
  --diarize \
  --samples ./additional-reference-samples
```

A pre-labelled script can be supplied instead:

```json
{
  "language": "en",
  "segments": [
    {"start": 0.4, "end": 2.8, "speaker": "ALICE", "text": "Hello."},
    {"start": 3.1, "end": 5.0, "speaker": "BOB", "text": "Welcome."}
  ]
}
```

## Complete run

```sh
synapse-dub-runtime setup
synapse-dub doctor

synapse-dub run source.mp4 \
  --workspace dubbing-work \
  --diarize \
  --samples ./samples \
  --target-language it \
  --backend wav2lip \
  --output dubbed-it.mp4
```

Translation is skipped when `--target-language` is omitted. The default endpoint
is configurable in `/etc/synapse/dub/config.conf`.

For unattended operation, prepare and manually review a map first, then pass it
with `--speaker-map FILE`. A non-interactive invocation without a confirmed map
is rejected.

## Runtime policy

The pacman package does not include FFmpeg, Python, PyTorch, ROCm, OpenCV,
Whisper, NumPy, or other libraries available as CachyOS packages. They are
normal `depends=()` entries. The compiled ROCm TorchAudio runtime is supplied by
the separate `synapse-python-torchaudio-rocm` pacman package. `synapse-dub-runtime`
creates a user-owned venv with `--system-site-packages` and installs only
explicitly listed projects missing from CachyOS, always with
`--no-deps`; it never downloads compiled ROCm components.

The Wav2Lip source tree is pinned and installed as an adapter, following the
same source-build approach used for DS4. Model weights are external.

MuseTalk and LatentSync were tested but intentionally omitted from release one.
MuseTalk requires the old compiled MMEngine/MMCV/MMPose stack, unavailable for
Python 3.14/system ROCm. LatentSync requires InsightFace plus ONNX Runtime, for
which no CPython 3.14 wheel or CachyOS Python package exists. Neither adapter
may create a second CUDA/PyTorch environment; they can be enabled in a later
release after those upstream stacks support the system runtime.

## Licenses

- XTTS-v2 model and outputs: Coqui Public Model License, non-commercial.
- Original Wav2Lip pretrained model: personal/research/non-commercial.
The package defaults to refusing XTTS-v2 until
`ACCEPT_NONCOMMERCIAL_MODEL_LICENSES=yes` is explicitly configured.
