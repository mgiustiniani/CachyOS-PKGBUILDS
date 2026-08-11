# Synapse ComfyUI ROCm

Pinned ComfyUI runtime for Radeon ROCm on Synapse Linux. The package contains
application code only; checkpoints, diffusion models, encoders, VAEs, LoRAs,
and outputs remain external.

```sh
sudo pacman -Syu
sudo reboot
synapse-rocm-doctor --pretty
synapse-comfyui-doctor
systemctl --user enable --now synapse-comfyui.service
```

Open `http://127.0.0.1:8188`. Configure the external model root in
`/etc/synapse/comfyui/config.conf`.

## Native MiniMax H3 support

This package is pinned after ComfyUI's native MiniMax H3 integration and H3 VAE
memory optimization. The bundled workflow-template registry includes native
T2VA, I2VA, R2VA, and API graphs. A user-editable native T2VA workflow is copied
to:

```text
~/.local/share/synapse/comfyui/user/default/workflows/synapse-minimax-h3-native-t2v.json
```

No H3 model weights or automatic H3 downloader are shipped. MiniMax H3 uses a
territorial community license that excludes the EU, UK, Republic of Korea, and
USA unless separate authorization is obtained. Users must review the upstream
license before acquiring or running the model.

## Managed Z-Image Turbo bundle

The packaged Z-Image Turbo manifest pins the Comfy-Org split-file revision,
exact sizes, and SHA-256 hashes for the BF16 model, Qwen3 4B encoder, and VAE:

```sh
sudo synapse-comfyui-models enqueue --source web --mode copy --priority 20
```

Web acquisition uses the official Hugging Face/Xet transport so multi-gigabyte
partials survive CDN changes. The launcher resolves copied or externally
activated components through `synapse-model resolve --json` and installs a
user-editable workflow template on first start.

The canonical archive remains format-first under
`safetensors/image-generation/comfyui/z-image-turbo`. No model weights are
included in this package or the ISO. Upstream Z-Image Turbo is Apache-2.0;
weights remain governed by their upstream terms. Community custom nodes execute
arbitrary Python code and are not installed automatically.

A headless Z-Image API prompt is available at:

```text
/usr/share/synapse/comfyui/workflows/z-image-turbo-api.json
```

ComfyUI image and video diffusion runs on Radeon ROCm. AMD Ryzen AI CVML is a
separate, license-gated C++ perception runtime for face detection, face mesh,
and depth estimation; it is not a general ComfyUI NPU backend.
