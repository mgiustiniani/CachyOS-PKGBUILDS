# Synapse Translation ROCm

Local specialist speech/text translation server using SeamlessM4T-v2-large and
system ROCm PyTorch. It binds to `127.0.0.1:8091` by default.

Endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/text/translations`
- `POST /v1/audio/translations` (multipart audio)

Example:

```sh
curl http://127.0.0.1:8091/v1/text/translations \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Local AI needs VRAM."],"source_language":"eng","target_language":"ita"}'
```

Models remain external to packages and ISO images. Internet provisioning:

```sh
sudo synapse-translate-models download
sudo systemctl enable --now synapse-translate.service
```

The shared `synapse-python-hf-compat` package supplies pinned Transformers,
Tokenizers and SentencePiece. The server uses `python-pytorch-opt-rocm`; it does
not create a venv or install another PyTorch copy.
