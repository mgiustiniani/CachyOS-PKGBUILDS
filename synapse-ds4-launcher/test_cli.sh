#!/bin/sh
set -eu

launcher=${1:-./ds4-launcher}
case "$launcher" in
    /*) ;;
    *) launcher=$(cd "$(dirname "$launcher")" && pwd)/$(basename "$launcher") ;;
esac

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ds4-launcher-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp/ds4" "$tmp/home" "$tmp/profile-work"

for executable in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
    cat >"$tmp/ds4/$executable" <<'SCRIPT'
#!/bin/sh
{
    printf 'executable=%s\n' "$(basename "$0")"
    printf 'pwd=%s\n' "$PWD"
    printf 'guard=%s\n' "${DS4_GLM_MEMORY_GUARD-}"
    for argument in "$@"; do printf 'arg=%s\n' "$argument"; done
} >"$DS4_TEST_OUTPUT"
SCRIPT
    chmod +x "$tmp/ds4/$executable"
done

ln -s "$tmp/ds4" "$tmp/home/ds4-rocm-0731"
mkdir -p "$tmp/home/code"
ln -s "$tmp/ds4" "$tmp/home/code/ds4"

strix_profile="$(dirname "$launcher")/profiles/deepseek-v4-flash-0731-strix.conf"
if [ -f "$strix_profile" ]; then
    strix_fast=$(HOME="$tmp/home" "$launcher" ds4-server \
        --profiles "$strix_profile" --profile deepseek-v4-flash-0731-strix \
        --no-prompt --dry-run)
    printf '%s\n' "$strix_fast" | grep -q "$tmp/ds4/ds4-server"
    printf '%s\n' "$strix_fast" | grep -q -- '--backend rocm'
    printf '%s\n' "$strix_fast" | grep -q -- '--prefill-chunk 8192'
    printf '%s\n' "$strix_fast" | grep -q -- '-c 131072 -n 30000 --host 0.0.0.0 --port 8000'
    printf '%s\n' "$strix_fast" | grep -q 'deepseek-v4-flash-0731-fast-server-kv'
    if printf '%s\n' "$strix_fast" | grep -Eq 'DS4_CUDA_NO_Q8_F16_CACHE|DS4_ROCM_ATTN_DISABLE_PACKED_F16'; then
        echo 'fast Strix profile unexpectedly enabled a long-safe rollback' >&2
        exit 1
    fi

    strix_long=$(HOME="$tmp/home" "$launcher" ds4-server \
        --profiles "$strix_profile" --profile deepseek-v4-flash-0731-strix-long-safe \
        --no-prompt --dry-run)
    printf '%s\n' "$strix_long" | grep -q 'DS4_CUDA_NO_Q8_F16_CACHE=1'
    if printf '%s\n' "$strix_long" | grep -q 'DS4_ROCM_ATTN_DISABLE_PACKED_F16'; then
        echo 'Strix long-safe bypassed the engine cache-aware attention gate' >&2
        exit 1
    fi
    printf '%s\n' "$strix_long" | grep -q -- '-c 240000 -n 30000 --host 0.0.0.0 --port 8000'
    printf '%s\n' "$strix_long" | grep -q 'deepseek-v4-flash-0731-long-safe-server-kv'

    strix_agent=$(HOME="$tmp/home" "$launcher" ds4-agent \
        --profiles "$strix_profile" --profile deepseek-v4-flash-0731-strix-long-safe \
        --no-prompt --dry-run)
    if printf '%s\n' "$strix_agent" | grep -q -- '--backend rocm'; then
        echo 'Strix agent received the unsupported explicit ROCm backend' >&2
        exit 1
    fi
    if printf '%s\n' "$strix_long" | grep -Eq 'DS4_ROCM_MOE_DISABLE_RAW_STAGE|DS4_ROCM_Q8_TRANSPOSE_SCALAR|--warm-weights'; then
        echo 'bundled Strix profile enabled a rollback or warm-weights option' >&2
        exit 1
    fi
    for power_executable in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
        for power_profile in deepseek-v4-flash-0731-strix deepseek-v4-flash-0731-strix-long-safe; do
            power_preview=$(HOME="$tmp/home" "$launcher" "$power_executable" \
                --profiles "$strix_profile" --profile "$power_profile" \
                --no-prompt --dry-run)
            printf '%s\n' "$power_preview" | grep -q -- '--power 100'
        done
    done
fi

dgx_profile="$(dirname "$launcher")/profiles/deepseek-v4-flash-0731-dgx.conf"
if [ -f "$dgx_profile" ]; then
    dgx_fast=$(HOME="$tmp/home" "$launcher" ds4-server \
        --profiles "$dgx_profile" --profile deepseek-v4-flash-0731-dgx \
        --no-prompt --dry-run)
    printf '%s\n' "$dgx_fast" | grep -q "$tmp/ds4/ds4-server"
    printf '%s\n' "$dgx_fast" | grep -q -- '--cuda'
    printf '%s\n' "$dgx_fast" | grep -q -- '-c 131072 -n 30000 --host 0.0.0.0 --port 8000'
    if printf '%s\n' "$dgx_fast" | grep -Eq 'DS4_CUDA_NO_Q8_F16_CACHE|--prefill-chunk'; then
        echo 'fast DGX profile has a long-safe or unsupported chunk option' >&2
        exit 1
    fi

    dgx_long=$(HOME="$tmp/home" "$launcher" ds4-server \
        --profiles "$dgx_profile" --profile deepseek-v4-flash-0731-dgx-long-safe \
        --no-prompt --dry-run)
    printf '%s\n' "$dgx_long" | grep -q 'DS4_CUDA_NO_Q8_F16_CACHE=1'
    printf '%s\n' "$dgx_long" | grep -q 'DS4_SERVER_COALESCE=0'
    printf '%s\n' "$dgx_long" | grep -q 'DS4_METAL_PREFILL_CHUNK=2048'
    if printf '%s\n' "$dgx_long" | grep -q 'DS4_ROCM_ATTN_DISABLE_PACKED_F16'; then
        echo 'DGX profile unexpectedly received a ROCm-only rollback' >&2
        exit 1
    fi
    printf '%s\n' "$dgx_long" | grep -q -- '-c 300000 -n 30000 --host 0.0.0.0 --port 8000'
    printf '%s\n' "$dgx_long" | grep -q 'deepseek-v4-flash-0731-dgx-session-safe-300k-server-kv'

    dgx_agent=$(HOME="$tmp/home" "$launcher" ds4-agent \
        --profiles "$dgx_profile" --profile deepseek-v4-flash-0731-dgx-long-safe \
        --no-prompt --dry-run)
    printf '%s\n' "$dgx_agent" | grep -q -- '--cuda -c 300000 -n 30000'
    printf '%s\n' "$dgx_agent" | grep -q 'DS4_METAL_PREFILL_CHUNK=2048'
    for power_executable in ds4 ds4-server ds4-bench ds4-eval ds4-agent; do
        for power_profile in deepseek-v4-flash-0731-dgx deepseek-v4-flash-0731-dgx-long-safe; do
            power_preview=$(HOME="$tmp/home" "$launcher" "$power_executable" \
                --profiles "$dgx_profile" --profile "$power_profile" \
                --no-prompt --dry-run)
            printf '%s\n' "$power_preview" | grep -q -- '--power 100'
        done
    done
fi

HOME="$tmp/home" "$launcher" --list | grep -q '^ds4-bench:'
HOME="$tmp/home" "$launcher" -D "$tmp/ds4" --list-executables |
    grep -q "ds4-agent.*$tmp/ds4/ds4-agent"

preview=$(HOME="$tmp/home" "$launcher" -D "$tmp/ds4" \
    --profile glm-worker-strix --no-prompt --dry-run \
    --set COORDINATOR_IP=10.44.0.9)
printf '%s\n' "$preview" | grep -q -- '--coordinator 10.44.0.9 19800'
printf '%s\n' "$preview" | grep -q 'DS4_GLM_MEMORY_GUARD=0'

server_preview=$(HOME="$tmp/home" "$launcher" ds4-server -D "$tmp/ds4" \
    --profile glm52-dgx-strix --no-prompt --dry-run \
    --set LISTEN_IP=10.44.0.77 --set HOST=0.0.0.0 --set HTTP_PORT=18000)
printf '%s\n' "$server_preview" | grep -q '/ds4-server '
printf '%s\n' "$server_preview" | grep -q -- '--listen 10.44.0.77 19800'
printf '%s\n' "$server_preview" | grep -q -- '--host 0.0.0.0 --port 18000'
if printf '%s\n' "$server_preview" | grep -q -- '--ctx-start'; then
    echo 'ds4-server received ds4-bench scoped arguments' >&2
    exit 1
fi
if HOME="$tmp/home" "$launcher" -D "$tmp/ds4" --profile glm52-dgx-strix \
    --no-prompt --dry-run >/dev/null 2>&1; then
    echo 'multi-executable profile unexpectedly omitted --executable' >&2
    exit 1
fi

DS4_TEST_OUTPUT="$tmp/output" HOME="$tmp/home" "$launcher" -D "$tmp/ds4" \
    --profile local-one-shot --no-prompt --yes \
    --set PROMPT='hello from profile' --set TOKENS=17 >/dev/null
grep -q '^executable=ds4$' "$tmp/output"
grep -q "^pwd=$tmp/ds4$" "$tmp/output"
grep -q '^arg=hello from profile$' "$tmp/output"
grep -q '^arg=17$' "$tmp/output"

cat >"$tmp/custom.conf" <<'PROFILE'
[profile custom-worker]
name=Custom worker
executable=ds4
description=Test custom profile.
variable=IP|Coordinator IP|192.0.2.10
variable=PORT|Port|1234
environment=DS4_GLM_MEMORY_GUARD=0
argument=--coordinator
argument=${IP}
argument=${PORT}

[profile bound-local]
name=Profile-bound DS4 tree
executable=ds4
description=Test profile-specific executable and working directories.
ds4_dir=${HOME}/../ds4
workdir=${HOME}/../profile-work
argument=profile-bound
PROFILE
custom=$(HOME="$tmp/home" "$launcher" -D "$tmp/ds4" --profiles "$tmp/custom.conf" \
    --profile custom-worker --no-prompt --dry-run --set IP=192.0.2.99)
printf '%s\n' "$custom" | grep -q -- '--coordinator 192.0.2.99 1234'

DS4_TEST_OUTPUT="$tmp/output" HOME="$tmp/home" "$launcher" \
    --profiles "$tmp/custom.conf" --profile bound-local --no-prompt --yes >/dev/null
grep -q '^executable=ds4$' "$tmp/output"
grep -q "^pwd=$tmp/profile-work$" "$tmp/output"
grep -q '^arg=profile-bound$' "$tmp/output"

if HOME="$tmp/home" "$launcher" -D "$tmp/ds4" --profile custom-worker \
    --profiles "$tmp/custom.conf" --no-prompt --dry-run --set MISSING=x \
    >/dev/null 2>&1; then
    echo 'unknown variable override unexpectedly succeeded' >&2
    exit 1
fi

echo 'ds4-launcher tests: PASS'
