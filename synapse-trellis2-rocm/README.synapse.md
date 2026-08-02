# synapse-trellis2-rocm

Synapse/CachyOS integration package draft for TRELLIS.2 ROCm on AMD Strix Halo (`gfx1151`).

## Current status

Manual validation on `paperboy@192.168.5.157` succeeded outside pacman:

- PyTorch ROCm `2.12.1`, HIP `7.2.53211`, device `AMD Radeon 8060S Graphics`.
- ROCm extension imports work in the manual venv:
  - `cumesh`
  - `flex_gemm`
  - `o_voxel`
  - `nvdiffrast.torch`
  - `nvdiffrec_render`
- TRELLIS.2 512 run succeeded on the better `cand_01` Scarlett input.
- GLB export succeeded.

Validated output:

```text
/home/paperboy/trellis2-rocm-test/outputs/scarlett_cand01_512_quality.glb
```

## Package scope of this draft

This PKGBUILD packages:

- ROCm base tree from `Cardboard-box-a/TRELLIS.2_rocm` pinned at `ee7be9e7b646eaefc761f9efc47ac96e7b9ee6ce`.
- The Microsoft tree is not packaged as the runtime base; it remains only the upstream reference/audit baseline.
- Local runtime patches for configurable DINOv3 and RMBG/BiRefNet sources.
- Runtime environment script.
- Model downloader.
- Preprocess helper.
- Image-to-GLB runner.
- Editable model source config.
- Non-prefixed user commands: `trellis2-download-models`, `trellis2-preprocess`, `trellis2-image-to-3d`.

## Bundled runtime pieces

To avoid a chain of moving Arch `*-git` dependency packages, this package draft vendors the missing Python runtime pieces under:

```text
/usr/lib/synapse-trellis2-rocm/python
```

Vendored Python deps:

```text
easydict
gradio==6.0.1
kornia
kornia-rs
timm
plyfile
pyfqmr
lpips
einops
moderngl
glcontext
utils3d @ 9a4eb15e4021b67b12c460c7057d642626897ec8
transformers==4.56.0
tokenizers==0.22.2
huggingface-hub==0.36.2

The PKGBUILD installs these with `pip --no-deps` so pip cannot replace the CachyOS ROCm PyTorch stack with upstream CPU/CUDA torch wheels. `tokenizers` is vendored because the current repo `python-tokenizers` version is newer than the version accepted by the validated Transformers stack.
```

Vendored ROCm/HIP extensions, built in the PKGBUILD:

```text
CuMesh @ 2ef5f9d121e460d1aca9fd8cadb9264d6fa94ba4
  - cubvh submodule @ ce92267a24ef6ad7d2c8ccbc2ae2c021a6597e70
FlexGEMM-rocm @ 2a92675f85027f66c7108381a5c0312b7b393310
o-voxel @ c4c8505c624c0e13a730ff1fd16e84f88837eb77
nvdiffrast-hip @ d29892337190f7e55f891b319d1daa6686bb681c
nvdiffrec @ a762dedf4aa049ce777ef8f453098d95f701e6c1
```

The next validation step is to build this package on the Strix Halo host and verify the packaged imports outside the manual venv.

## Model download policy

Weights are not bundled in package or ISO, but installation must download them into an external cache.

Default config:

```text
/etc/synapse/trellis2-rocm/models.conf
```

Default cache:

```text
/var/lib/synapse/trellis2/models
```

Installer download command used by the package/install flow:

```bash
trellis2-download-models
```

There is intentionally no separate `setup` executable: setup is part of installation. The config deliberately exposes model URLs so admins can replace upstreams with mirrors:

```text
TRELLIS2_MODEL_URL=
TRELLIS_IMAGE_LARGE_URL=
DINO_MODEL_URL=
REMBG_MODEL_URL=
```

This is required because DINOv3 and RMBG upstream models can be gated, moved, or removed.

## Runtime commands

Preprocess image:

```bash
trellis2-preprocess input.jpg --out input_preprocessed.png
```

Generate GLB:

```bash
trellis2-image-to-3d input_preprocessed.png \
  --out output.glb \
  --pipeline-type 512 \
  --steps 12 \
  --decimation-target 200000 \
  --texture-size 1024
```

Do not pass `--preprocess` if the input is already preprocessed.

## Background removal caveat

Current CachyOS `python-torchvision` lacks a HIP kernel for `torchvision::deform_conv2d`, used by BiRefNet. The validated workaround is:

```text
REMBG_FORCE_CPU=1
REMBG_MODEL_NAME=ZhengPeng7/BiRefNet
```

This is the default in the packaged config.
