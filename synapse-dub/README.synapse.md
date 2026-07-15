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

The package contains its complete runtime. Calamares Server Setup downloads or
validates the external models after license acceptance. Verify the installation:

```sh
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

## Completely custom scene from a real video

The `custom` flow keeps the source footage, timing, and speaker sequence while an
OpenAI-compatible LLM writes entirely new timed dialogue from a creative brief.
The generated editable manifest is saved as `WORKSPACE/dialogue-custom.json`.
Voice mapping remains manual and mandatory before synthesis.

```sh
synapse-dub custom real-video.mp4 \
  --workspace custom-work \
  --diarize \
  --samples ./samples \
  --prompt "Turn this scene into an ironic discussion about open source." \
  --target-language it \
  --backend wav2lip \
  --output custom-video-it.mp4
```

For unattended operation, prepare and manually review a map first, then pass it
with `--speaker-map FILE`. A non-interactive invocation without a confirmed map
is rejected.

## Fully generative visual production

Install and provision the shared Wan2.1 VACE ROCm backend, normally through
Calamares Server Setup, then run:

```sh
synapse-dub generative source.mp4 \
  --workspace generative-work \
  --visual-prompt 'A robot detective crosses a rainy neon city, cinematic tracking shot' \
  --dialogue-prompt 'Write a tense exchange about a missing artificial intelligence' \
  --target-language it \
  --backend wav2lip \
  --output generative-final.mp4
```

This creates new visual frames rather than preserving the source footage. The
source contributes timing, speech, voices, and a reference frame for each
five-second generated shot. Wav2Lip modifies frames containing a detected face
and preserves frames without one; use `--backend none` when lip synchronization
is not appropriate. Generation parameters and the pinned model revision are recorded
in `generative-visual.generation.json`.

## Translate an authorized YouTube video

The repository `yt-dlp` package downloads one video; playlists are disabled.
Only download content you own or are authorized to process. YouTube translation
preserves the original video stream by default and replaces only its audio.
Pass `--backend wav2lip` explicitly when mouth-region modification is desired.

```sh
synapse-dub youtube 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --workspace youtube-it-work \
  --diarize \
  --target-language it \
  --output youtube-it.mp4
```

For an owned video requiring authentication, pass `--cookies FILE` or
`--cookies-from-browser BROWSER`. The downloaded source and yt-dlp metadata stay
inside the selected workspace for reproducibility.

The complete implemented and planned workflow inventory is installed at
`/usr/share/doc/synapse-dub/WORKFLOWS.md`.

## Runtime policy

The pacman package does not include FFmpeg, Python, PyTorch, ROCm, OpenCV,
Whisper, NumPy, or other libraries available as CachyOS packages. They are
normal `depends=()` entries. The compiled ROCm TorchAudio runtime is supplied by
the separate `synapse-python-torchaudio-rocm` pacman package. Projects unavailable
from CachyOS are pinned and installed into `/usr/lib/synapse-dub/python` at package
build time. No runtime installer or user venv downloads compiled ROCm components.

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
