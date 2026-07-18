# Synapse ComfyUI ROCm

Pinned ComfyUI runtime for Radeon ROCm on Synapse Linux. The package contains
application code only; checkpoints, diffusion models, encoders, VAEs, LoRAs,
and outputs remain external.

```sh
synapse-comfyui-doctor
sudo synapse-comfyui-models enqueue --source web --mode copy --priority 20
systemctl --user enable --now synapse-comfyui.service
```

Open `http://127.0.0.1:8188`. Configure the external model root in
`/etc/synapse/comfyui/config.conf`. The packaged Z-Image Turbo manifest pins the
Comfy-Org split-file revision, exact sizes, and SHA-256 hashes for the BF16
model, Qwen3 4B encoder, and VAE. Web acquisition uses the official Hugging
Face/Xet transport so multi-gigabyte partials survive CDN changes. The launcher resolves either copied or
externally activated model components through `synapse-model resolve --json`
and installs a user-editable workflow template on first start.

The canonical archive remains format-first under
`safetensors/image-generation/comfyui/z-image-turbo`. No model weights are
included in this package or the ISO. Upstream Z-Image Turbo is Apache-2.0;
weights remain governed by their upstream terms. Community custom nodes execute
arbitrary Python code and are not installed automatically.

A headless API prompt is available at:

```text
/usr/share/synapse/comfyui/workflows/z-image-turbo-api.json
```

ComfyUI image and video diffusion runs on Radeon ROCm. AMD Ryzen AI CVML is a
separate, license-gated C++ perception runtime for face detection, face mesh,
and depth estimation; it is not a general ComfyUI NPU backend.
