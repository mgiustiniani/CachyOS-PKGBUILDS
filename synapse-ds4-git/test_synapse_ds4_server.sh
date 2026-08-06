#!/bin/sh
set -eu

wrapper=$1
tmp=$(mktemp -d "${TMPDIR:-/tmp}/synapse-ds4-server-test.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

cat > "$tmp/launcher" <<'SCRIPT'
#!/bin/sh
for argument in "$@"; do
    printf '%s\n' "$argument"
done
SCRIPT
chmod +x "$tmp/launcher"

cat > "$tmp/config" <<EOF
MODEL_PATH=$tmp/model.gguf
MTP_PATH=
MTP_DRAFT_TOKENS=2
POWER=100
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
SERVER_CTX=100000
SERVER_TOKENS=30000
SERVER_KV_DIR=/var/lib/ds4/kv/server
SERVER_KV_MB=8192
EOF

output=$(
    DS4_CONFIG="$tmp/config" \
    DS4_LAUNCHER="$tmp/launcher" \
    DS4_LAUNCHER_PROFILES=/tmp/system.profile \
    bash "$wrapper" --dry-run
)

printf '%s\n' "$output" | grep -Fxq 'ds4-server'
printf '%s\n' "$output" | grep -Fxq 'synapse-ds4-server'
printf '%s\n' "$output" | grep -Fxq 'MODEL='"$tmp"'/model.gguf'
printf '%s\n' "$output" | grep -Fxq 'HOST=127.0.0.1'
printf '%s\n' "$output" | grep -Fxq 'TOKENS=30000'
printf '%s\n' "$output" | grep -Fxq -- '--profiles'
printf '%s\n' "$output" | grep -Fxq '/tmp/system.profile'
printf '%s\n' "$output" | grep -Fxq -- '--dry-run'

if DS4_CONFIG="$tmp/config" DS4_LAUNCHER="$tmp/launcher" bash "$wrapper" >/dev/null 2>&1; then
    echo 'wrapper unexpectedly accepted a missing model for a live run' >&2
    exit 1
fi

printf '%s\n' 'synapse-ds4-server wrapper tests: PASS'
