# Synapse Voice AI ROCm

Local multilingual speech recognition and voice-cloning TTS for Synapse Linux.

## Components

- Chatterbox Multilingual V2, including Italian and zero-shot voice cloning
- whisper.cpp ROCm with Whisper large-v3-turbo Q8
- OpenAI-compatible speech and transcription endpoints
- user-scoped voice enrollment

## Models

Weights are not included in the package or ISO. The default model archive uses
format-first categories:

- `safetensors/voice-ai/huggingface/ResembleAI/chatterbox`
- `ggml/speech-recognition/whisper.cpp/ggml-large-v3-turbo-q8_0.bin`

Mount the filesystem labelled `models`, then validate it:

```sh
synapse-voice-import-models
```

Copy models into the installed system if desired:

```sh
sudo synapse-voice-import-models --copy
```

## Voice enrollment

```sh
synapse-voice-enroll my-voice reference.wav
synapse-voice-tts --voice my-voice --language it -o result.wav \
  "Questo testo utilizza la voce campionata."
```

Reference audio remains under the current user's XDG data directory with private
permissions.

## API

```sh
systemctl --user enable --now synapse-voice-api.service
curl http://127.0.0.1:8090/health
```

The API implements `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`,
`POST /v1/voice/chat`, and `GET /v1/voices`. The voice-chat endpoint performs
ASR → OpenAI-compatible LLM → cloned-voice TTS and returns JSON with WAV audio
encoded as base64.
