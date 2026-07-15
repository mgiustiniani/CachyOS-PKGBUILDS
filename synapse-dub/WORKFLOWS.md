# Synapse Dub workflow inventory

## Available

### 1. Calamares Server provisioning

ROCm profile → Server Setup license acceptance → package installation → Internet
model download or existing-directory validation → `synapse-dub doctor`.

### 2. Source-video preparation

`synapse-dub prepare` extracts audio, transcribes it, optionally diarizes speakers,
extracts candidate voice references, and requires explicit speaker-to-sample
mapping. The workspace remains editable and reusable.

### 3. Translation dubbing

`synapse-dub run VIDEO --target-language LANG` translates the timed source
transcript through an OpenAI-compatible endpoint, clones the confirmed voices,
rebuilds the audio timeline, applies Wav2Lip, and muxes the result.

### 4. Pre-labelled production script

`prepare` and `run` accept `--script dialogue.json`. This bypasses Whisper text
generation while preserving mandatory sample mapping. A reviewed
`--speaker-map` supports reproducible non-interactive production.

### 5. Creative re-dubbing of real footage

`synapse-dub custom VIDEO --prompt BRIEF` asks an OpenAI-compatible LLM to write
new timed dialogue while preserving source timestamps and speaker order. It saves
`dialogue-custom.json`, then performs XTTS-v2 synthesis and Wav2Lip. This changes
dialogue, voices, and mouth movement; it does not generate new visual scenes.

### 6. YouTube translation

`synapse-dub youtube URL --target-language LANG` uses the repository `yt-dlp`
package to download one authorized video, records its source metadata, and runs
the normal translation-dubbing pipeline. Playlists are intentionally disabled.

### 7. External voice samples and source-video references

Every preparation flow can combine `--samples FILE_OR_DIRECTORY` with clean
candidate references extracted from the source video. No detected identity is
silently assigned to a voice.

### 8. Automatic diarization

`--diarize` uses the configured pyannote model and `HUGGINGFACE_TOKEN`. A
pre-labelled script remains available when credentials or model access are not
available.

## Not implemented

### Generative visual video-to-video

A workflow that creates new frames, actions, backgrounds, camera motion, or
characters from real footage is not implemented. It requires a separately
packaged ROCm video-to-video backend and must precede the existing dialogue,
voice-cloning, lip-sync, and mux stages.
