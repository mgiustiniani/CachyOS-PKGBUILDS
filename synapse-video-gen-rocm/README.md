# Synapse Video Gen ROCm

Shared, install-ready generative visual backend for Synapse products. It uses
Wan2.1 VACE 1.3B through system ROCm PyTorch, an application-private Diffusers
layer, and the shared `synapse-python-hf-compat` runtime. It never creates a
venv and does not duplicate PyTorch or compiled Hugging Face components.

Model weights are not part of the package or ISO. Provision the pinned
Apache-2.0 model with:

```sh
sudo synapse-video-gen-models download
synapse-video-gen doctor
```

Generate a genuinely new visual clip using the source video's frame at the
start of each shot as a visual reference:

```sh
synapse-video-gen generate \
  --input source.mp4 \
  --prompt 'A cinematic robot detective crosses a rainy neon city' \
  --output generated.mp4
```

The backend generates 480p, 16 fps shots by default and supports source videos
up to 60 seconds. Long inputs are divided into five-second shots while the
pipeline remains loaded. A `*.generation.json` provenance manifest records the
model revision and all generation parameters.

The real source supplies visual reference frames, not its original motion.
Consequently this mode can create new scenes and actions but may introduce
identity changes or discontinuities between generated shots.
