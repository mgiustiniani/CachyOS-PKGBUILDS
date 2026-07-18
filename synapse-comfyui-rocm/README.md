# Synapse ComfyUI ROCm

Pinned ComfyUI runtime for Radeon ROCm on Synapse Linux. The package contains
application code only; checkpoints, diffusion models, encoders, VAEs, LoRAs,
and outputs remain external.

```sh
synapse-comfyui-doctor
systemctl --user enable --now synapse-comfyui.service
```

Open `http://127.0.0.1:8188`. Configure the external model root in
`/etc/synapse/comfyui/config.conf`. Community custom nodes execute arbitrary
Python code and are not installed automatically.

ComfyUI image and video diffusion runs on Radeon ROCm. AMD Ryzen AI CVML is a
separate, license-gated C++ perception runtime for face detection, face mesh,
and depth estimation; it is not a general ComfyUI NPU backend.
