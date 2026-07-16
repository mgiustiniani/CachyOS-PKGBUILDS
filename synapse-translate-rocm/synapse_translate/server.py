from __future__ import annotations

import argparse
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from scipy.signal import resample_poly

CONFIG_PATH = Path("/etc/synapse/translate.conf")
MODEL_REVISION = "5f8cc790b19fc3f67a61c105133b20b34e3dcb76"
LANGUAGES = {
    "de": "deu", "deu": "deu", "en": "eng", "eng": "eng",
    "es": "spa", "spa": "spa", "fr": "fra", "fra": "fra",
    "it": "ita", "ita": "ita", "ja": "jpn", "jpn": "jpn",
    "ko": "kor", "kor": "kor", "pt": "por", "por": "por",
    "ru": "rus", "rus": "rus", "zh": "cmn", "cmn": "cmn",
}


def load_config() -> dict[str, str]:
    values: dict[str, str] = {}
    if CONFIG_PATH.is_file():
        for raw in CONFIG_PATH.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def language_code(value: str) -> str:
    code = value.strip().lower().replace("_", "-").split("-", 1)[0]
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language code: {value}")
    return LANGUAGES[code]


def validate_model(root: Path) -> list[str]:
    required = (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "tokenizer_config.json", "tokenizer.model",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if len(list(root.glob("model-*.safetensors"))) < 2:
        missing.append("model-*.safetensors")
    return missing


class TextTranslationRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=128)
    source_language: str = "eng"
    target_language: str


class TranslationBackend:
    def __init__(self, model_root: Path, dtype_name: str = "bfloat16") -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        missing = validate_model(model_root)
        if missing:
            raise RuntimeError("Incomplete SeamlessM4T model: " + ", ".join(missing))
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch ROCm does not expose the AMD GPU through torch.cuda")
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
        self.model = SeamlessM4Tv2Model.from_pretrained(
            model_root, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
        ).to("cuda")
        self.model.eval()
        self.lock = threading.Lock()
        self.model_root = model_root

    @staticmethod
    def _sequences(generated: Any):
        if isinstance(generated, torch.Tensor):
            return generated
        if hasattr(generated, "sequences"):
            return generated.sequences
        if isinstance(generated, (tuple, list)):
            return generated[0]
        raise RuntimeError(f"Unexpected Seamless generation result: {type(generated)!r}")

    def translate_text(self, texts: list[str], source: str, target: str) -> list[str]:
        source_code = language_code(source)
        target_code = language_code(target)
        inputs = self.processor(text=texts, src_lang=source_code, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with self.lock, torch.inference_mode():
            generated = self.model.generate(
                **inputs, tgt_lang=target_code, generate_speech=False, max_new_tokens=512
            )
        sequences = self._sequences(generated)
        return [text.strip() for text in self.processor.batch_decode(sequences, skip_special_tokens=True)]

    def translate_audio(self, audio: np.ndarray, sample_rate: int, source: str, target: str) -> str:
        source_code = language_code(source)
        target_code = language_code(target)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if sample_rate != 16000:
            audio = resample_poly(audio, 16000, sample_rate).astype(np.float32)
        inputs = self.processor(
            audio=audio, sampling_rate=16000, src_lang=source_code, return_tensors="pt"
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with self.lock, torch.inference_mode():
            generated = self.model.generate(
                **inputs, tgt_lang=target_code, generate_speech=False, max_new_tokens=512
            )
        sequences = self._sequences(generated)
        return self.processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()


def create_app() -> FastAPI:
    config = load_config()
    model_root = Path(config.get("MODEL_ROOT", "/var/lib/synapse/translate/models/seamless-m4t-v2-large"))
    backend = TranslationBackend(model_root, config.get("DTYPE", "bfloat16"))
    app = FastAPI(title="Synapse Translation Server", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok", "backend": "seamless-m4t-v2-large",
            "model_root": str(backend.model_root), "device": torch.cuda.get_device_name(0),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "seamless-m4t-v2-large", "object": "model", "revision": MODEL_REVISION}],
        }

    @app.post("/v1/text/translations")
    def text_translation(request: TextTranslationRequest) -> dict[str, Any]:
        try:
            translations = backend.translate_text(
                request.texts, request.source_language, request.target_language
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"translations": translations, "model": "seamless-m4t-v2-large"}

    @app.post("/v1/audio/translations")
    async def audio_translation(
        file: UploadFile = File(...), source_language: str = Form("eng"),
        target_language: str = Form(...),
    ) -> dict[str, Any]:
        try:
            audio, sample_rate = sf.read(io.BytesIO(await file.read()), always_2d=False)
            translation = backend.translate_audio(audio, sample_rate, source_language, target_language)
        except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"translation": translation, "model": "seamless-m4t-v2-large"}

    return app


def doctor(model_root: Path) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("PyTorch ROCm", torch.cuda.is_available(), torch.__version__))
    for module in ("transformers", "tokenizers", "sentencepiece", "fastapi", "uvicorn"):
        try:
            imported = __import__(module)
            checks.append((module, True, getattr(imported, "__version__", "available")))
        except Exception as exc:
            checks.append((module, False, str(exc)))
    missing = validate_model(model_root)
    checks.append(("SeamlessM4T model", not missing, str(model_root) if not missing else ", ".join(missing)))
    failed = False
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name:22} {detail}")
        failed |= not ok
    return int(failed)


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(prog="synapse-translate-server")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default=config.get("BIND", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(config.get("PORT", "8091")))
    check = sub.add_parser("doctor")
    check.add_argument(
        "--model-root", type=Path,
        default=Path(config.get("MODEL_ROOT", "/var/lib/synapse/translate/models/seamless-m4t-v2-large")),
    )
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor(args.model_root)
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
