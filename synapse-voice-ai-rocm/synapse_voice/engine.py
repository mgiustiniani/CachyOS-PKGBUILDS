from __future__ import annotations

import io
import threading
from pathlib import Path

import soundfile as sf

from .config import load_config, resolve_model_paths, voice_directory


class VoiceEngine:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._sample_rate = 24000

    def _load(self):
        if self._model is not None:
            return self._model

        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        cfg = load_config()
        chatterbox_path, _ = resolve_model_paths(cfg)
        requested = cfg.get("VOICE_DEVICE", "cuda")
        device = requested if requested != "cuda" or torch.cuda.is_available() else "cpu"
        self._model = ChatterboxMultilingualTTS.from_local(chatterbox_path, device)
        self._sample_rate = self._model.sr
        return self._model

    def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> bytes:
        if not text.strip():
            raise ValueError("Text must not be empty")
        cfg = load_config()
        language_id = language or cfg.get("VOICE_LANGUAGE", "it")
        sample = None
        if voice != "default":
            candidate = voice_directory() / f"{voice}.wav"
            if not candidate.is_file():
                raise FileNotFoundError(f"Unknown voice: {voice}")
            sample = str(candidate)

        with self._lock:
            model = self._load()
            audio = model.generate(
                text,
                language_id=language_id,
                audio_prompt_path=sample,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        array = audio.squeeze(0).detach().cpu().numpy()
        output = io.BytesIO()
        sf.write(output, array, self._sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()


engine = VoiceEngine()
