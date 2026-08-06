#!/bin/sh
set -eu

launcher=$1
profile=$2

tmp=$(mktemp -d "${TMPDIR:-/tmp}/synapse-ds4-launcher-profile.XXXXXX")
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

cat > "$tmp/ds4-server" <<'SCRIPT'
#!/bin/sh
exit 0
SCRIPT
chmod +x "$tmp/ds4-server"

preview=$(
    "$launcher" ds4-server \
        --ds4-dir "$tmp" \
        --profiles "$profile" \
        --profile synapse-ds4-server \
        --set MODEL=/var/lib/synapse/models/test.gguf \
        --set HOST=0.0.0.0 \
        --set HTTP_PORT=18000 \
        --set CTX=131072 \
        --set TOKENS=30000 \
        --set POWER=100 \
        --set KV_DIR=/var/lib/ds4/kv/test \
        --set KV_MB=4096 \
        --no-prompt \
        --dry-run
)

printf '%s\n' "$preview" | grep -Fq "$tmp/ds4-server"
printf '%s\n' "$preview" | grep -Fq -- '--backend rocm'
printf '%s\n' "$preview" | grep -Fq -- '-c 131072 -n 30000'
printf '%s\n' "$preview" | grep -Fq -- '--host 0.0.0.0 --port 18000'
printf '%s\n' "$preview" | grep -Fq -- '--power 100'
printf '%s\n' "$preview" | grep -Fq -- '--kv-disk-dir /var/lib/ds4/kv/test'
printf '%s\n' "$preview" | grep -Fq -- '--kv-disk-space-mb 4096 --cors'

printf '%s\n' 'synapse-ds4-launcher system profile tests: PASS'
