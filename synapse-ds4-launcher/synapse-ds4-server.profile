# System profile installed by synapse-ds4-launcher.
# Calamares and administrators override values with --set; no shell is used.

[profile synapse-ds4-server]
name=Synapse DS4 ROCm server
executable=ds4-server
description=Portable system DS4 server policy for Synapse Linux.
variable=MODEL|Verified DS4 GGUF model|/var/lib/synapse/models/gguf/deepseek4-flash-284B-GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
variable=HOST|HTTP bind address|127.0.0.1
variable=HTTP_PORT|HTTP port|8000
variable=CTX|Allocated context tokens|100000
variable=TOKENS|Maximum response tokens|30000
variable=POWER|GPU duty cycle percentage (1-100)|100
variable=KV_DIR|Persistent disk KV directory|/var/lib/ds4/kv/server
variable=KV_MB|Persistent disk KV budget MiB|8192
argument=-m
argument=${MODEL}
argument=--backend
argument=rocm
argument=-c
argument=${CTX}
argument=-n
argument=${TOKENS}
argument=--host
argument=${HOST}
argument=--port
argument=${HTTP_PORT}
argument=--power
argument=${POWER}
argument=--kv-disk-dir
argument=${KV_DIR}
argument=--kv-disk-space-mb
argument=${KV_MB}
argument=--cors
