from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .asr import Transcription, transcribe_wav
from .config import load_config, resolve_model_paths, voice_directory
from .engine import engine

app = FastAPI(title="Synapse Voice AI", version="0.2.0")


class SpeechRequest(BaseModel):
    model: str = "chatterbox-multilingual-v2"
    input: str = Field(min_length=1)
    voice: str = "default"
    language: str = "it"
    response_format: str = "wav"
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.5)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    try:
        chatterbox, whisper = resolve_model_paths()
        config = load_config()
        models = {"chatterbox": str(chatterbox), "whisper": str(whisper)}
        status = "ok"
    except FileNotFoundError as exc:
        models = {"error": str(exc)}
        config = load_config()
        status = "degraded"
    return {
        "status": status,
        "models": models,
        "whisper_backend": config.get("WHISPER_BACKEND", "rocm"),
        "xdna_available": shutil.which("synapse-whisper-xdna") is not None,
    }


@app.get("/v1/voices")
def voices() -> dict:
    names = ["default"]
    names.extend(sorted(path.stem for path in voice_directory().glob("*.wav")))
    return {"data": [{"id": name, "object": "voice"} for name in names]}


@app.post("/v1/audio/speech")
def speech(request: SpeechRequest) -> Response:
    if request.response_format != "wav":
        raise HTTPException(status_code=400, detail="Only WAV output is currently supported")
    try:
        payload = engine.synthesize(
            text=request.input,
            voice=request.voice,
            language=request.language,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(payload, media_type="audio/wav")


def _transcribe_audio(
    payload: bytes, filename: str | None, language: str, backend: str | None = None,
) -> Transcription:
    suffix = Path(filename or "audio").suffix[:12]
    with tempfile.TemporaryDirectory(prefix="synapse-voice-") as directory:
        root = Path(directory)
        source = root / f"source{suffix}"
        source.write_bytes(payload)
        wav = root / "input.wav"
        converted = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(source), "-ar", "16000", "-ac", "1", str(wav),
            ],
            capture_output=True,
            text=True,
        )
        if converted.returncode != 0:
            raise HTTPException(status_code=400, detail=converted.stderr.strip())
        try:
            return transcribe_wav(wav, language, backend=backend)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    language: str = Form(default="it"),
    backend: str | None = Form(default=None),
) -> dict:
    result = _transcribe_audio(await file.read(), file.filename, language, backend)
    return {"text": result.text, "backend": result.backend}


@app.post("/v1/voice/chat")
async def voice_chat(
    file: UploadFile = File(...),
    language: str = Form(default="it"),
    voice: str = Form(default="default"),
    model: str | None = Form(default=None),
    backend: str | None = Form(default=None),
) -> dict:
    transcription = _transcribe_audio(await file.read(), file.filename, language, backend)
    transcript = transcription.text
    if not transcript:
        raise HTTPException(status_code=400, detail="No speech was recognized")

    config = load_config()
    endpoint = config.get("VOICE_LLM_ENDPOINT", "http://192.168.5.157:8000/v1").rstrip("/")
    llm_model = model or config.get("VOICE_LLM_MODEL", "deepseek-v4-flash")
    try:
        response = httpx.post(
            f"{endpoint}/chat/completions",
            json={
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"Reply concisely in language code {language}.",
                    },
                    {"role": "user", "content": transcript},
                ],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    try:
        audio = engine.synthesize(answer, voice=voice, language=language)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "transcript": transcript,
        "transcription_backend": transcription.backend,
        "response": answer,
        "audio_format": "wav",
        "audio_base64": base64.b64encode(audio).decode("ascii"),
    }
