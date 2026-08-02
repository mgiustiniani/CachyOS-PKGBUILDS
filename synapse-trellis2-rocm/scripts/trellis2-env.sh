#!/usr/bin/env bash
# Runtime environment for packaged Synapse TRELLIS.2 ROCm.
set -euo pipefail

CONF="${TRELLIS2_MODELS_CONF:-/etc/synapse/trellis2-rocm/models.conf}"
if [[ -f "$CONF" ]]; then
  # shellcheck disable=SC1090
  source "$CONF"
fi

export TRELLIS2_APPDIR="${TRELLIS2_APPDIR:-/usr/lib/synapse-trellis2-rocm}"
export TRELLIS2_ROOT="${TRELLIS2_ROOT:-$TRELLIS2_APPDIR/TRELLIS.2}"
export TRELLIS2_PYTHON_VENDOR="${TRELLIS2_PYTHON_VENDOR:-$TRELLIS2_APPDIR/python}"
export TRELLIS2_MODEL_ROOT="${TRELLIS2_MODEL_ROOT:-/var/lib/synapse/trellis2/models}"
export HF_HOME="${HF_HOME:-$TRELLIS2_MODEL_ROOT/huggingface}"
export DINO_MODEL_PATH="${DINO_MODEL_PATH:-$TRELLIS2_MODEL_ROOT/dino/facebook/dinov3-vitl16-pretrain-lvd1689m}"
export REMBG_MODEL_NAME="${REMBG_MODEL_NAME:-ZhengPeng7/BiRefNet}"
export REMBG_FORCE_CPU="${REMBG_FORCE_CPU:-1}"

export GPU_ARCHS="${GPU_ARCHS:-gfx1151}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HSA_XNACK="${HSA_XNACK:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export FLEX_GEMM_ALGO="${FLEX_GEMM_ALGO:-explicit_gemm}"
export PYTHONPATH="$TRELLIS2_PYTHON_VENDOR:$TRELLIS2_ROOT:${PYTHONPATH:-}"
