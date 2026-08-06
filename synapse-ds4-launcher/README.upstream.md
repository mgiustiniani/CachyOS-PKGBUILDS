# DS4 Launcher

`ds4-launcher` is an external, dependency-free command-line profile launcher for
[DwarfStar/DS4](../ds4). It does not link to DS4 and does not modify DS4 source
code. The same C99/POSIX binary can be built on x86-64 Strix Halo systems and
ARM64 NVIDIA DGX Spark systems.

It solves three recurring problems:

1. discover and choose among `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and
   `ds4-agent`;
2. apply one tested hardware/model configuration to different DS4 executables;
3. review or change configuration values, including listen/coordinator IP
   addresses, for one execution without editing the profile.

The launcher uses `execv(2)` directly. It does not pass generated commands
through a shell.

## Build

```sh
cd ~/workspace/ds4-launcher
make
make test
```

Install if desired:

```sh
sudo make install PREFIX=/usr
```

Only a C compiler and POSIX libc are required. No CUDA or ROCm headers are
needed because the launcher remains independent from DS4.

## Interactive use

If DS4 is in `~/ds4`, the launcher discovers it automatically:

```sh
ds4-launcher
```

Otherwise provide its directory:

```sh
./ds4-launcher --ds4-dir ~/workspace/ds4
```

The launcher first shows only executable DS4 programs that actually exist. It
then shows every configuration compatible with the selected executable. A
multi-executable configuration can also be selected first; the launcher will
then ask which compatible executable to use. Every applicable variable is
displayed with its default value:

```text
Coordinator IP [10.44.0.1]:
Distributed control port [19800]:
```

Press Enter to confirm the profile default or type a replacement for this
execution. Values supplied immediately with `--set NAME=VALUE` are not asked
again; only missing values are confirmed afterward. A shell-quoted command
preview is printed before final execution confirmation.

## Non-interactive use

List profiles and detected executables:

```sh
ds4-launcher --list
ds4-launcher --list-executables --ds4-dir ~/ds4
```

Run the tested GLM 5.2 DGX Spark + Strix Halo configuration as `ds4-server`:

```sh
ds4-launcher ds4-server \
  --ds4-dir ~/ds4 \
  --profile glm52-dgx-strix
```

Preview the same configuration as `ds4-bench` without loading the model:

```sh
ds4-launcher ds4-bench \
  --ds4-dir ~/ds4 \
  --profile glm52-dgx-strix \
  --no-prompt \
  --dry-run
```

Override the default IP for one execution:

```sh
ds4-launcher \
  --ds4-dir ~/ds4 \
  --profile glm-worker-strix \
  --set COORDINATOR_IP=10.44.0.9 \
  --no-prompt \
  --yes
```

Append DS4 arguments after `--`:

```sh
ds4-launcher \
  --profile q4-benchmark-spark \
  --set LISTEN_IP=10.44.0.1 \
  --no-prompt \
  --dry-run \
  -- --dist-activation-bits 16
```

`DS4_DIR=/path/to/ds4` is equivalent to `--ds4-dir /path/to/ds4`.

## Built-in profiles

`glm52-dgx-strix` is a multi-executable configuration containing the validated
DGX Spark coordinator split `0:41`, Strix worker split `42:output`, CUDA cache
environment, model, and network defaults. It supports `ds4`, `ds4-server`,
`ds4-bench`, `ds4-eval`, and `ds4-agent`; only arguments and variables relevant
to the selected executable are added.

### `ds4`

- `local-chat`
- `local-one-shot`
- `glm-worker-strix`
- `q4-worker-strix`
- `q4-distributed-chat-spark`

### `ds4-server`

- `local-server`
- `q4-distributed-server-spark`

The HTTP bind address and distributed listen address are separate variables.
The HTTP default remains `127.0.0.1`.

### `ds4-bench`

- `local-benchmark`
- `glm-benchmark-spark`
- `q4-benchmark-spark`
- `q4-full-sweep-spark`

The distributed profiles reproduce the validated DGX Spark coordinator and
Strix Halo worker layer splits documented in
[`../ds4.md`](../ds4.md):

- GLM 5.2: Spark `0:41`, Strix `42:output`;
- DeepSeek V4 Flash Q4: Spark `0:21`, Strix `22:output`.

### `ds4-eval`

- `deterministic-eval`: documented four-question deterministic regression gate.

### `ds4-agent`

- `local-agent`

## Custom profiles

Profiles can be loaded from these locations, in order:

1. built-in profiles;
2. `/etc/ds4-launcher/profiles.conf`, when present;
3. `~/.config/ds4-launcher/profiles.conf`, when present;
4. the file passed through `--profiles FILE`.

A later profile with the same ID replaces the earlier profile. See
[`profiles.example.conf`](profiles.example.conf).

A configuration may support one executable with `executable=`, or several with
`executables=`. Common keys apply to all selected executables. A profile can
bind a specific DS4 build tree with `ds4_dir=` and a launch directory with
`workdir=`; `${HOME}` and profile variables are expanded. Explicit `--ds4-dir`
or `--workdir` options override those profile defaults. Append an executable
name to `variable.` or `argument.` for executable-specific values:

```ini
[profile lab-distributed]
name=Lab distributed configuration
executables=ds4,ds4-server,ds4-bench
description=One model/network configuration with three actions.
ds4_dir=${HOME}/ds4-lab
workdir=${HOME}/ds4-lab
variable=MODEL|GGUF model|gguf/model.gguf
variable=LISTEN_IP|Coordinator IP|10.44.0.1
variable=PORT|Distributed port|19800
variable.ds4-server=HOST|HTTP bind IP|127.0.0.1
argument=-m
argument=${MODEL}
argument=--listen
argument=${LISTEN_IP}
argument=${PORT}
argument.ds4-server=--host
argument.ds4-server=${HOST}
argument.ds4-bench=--ctx-start
argument.ds4-bench=2048
```

Each `argument` line becomes exactly one argument. Spaces are preserved and no
shell tokenization occurs. `${NAME}` resolves variables applicable to the
selected executable first and then process environment variables such as
`${HOME}`.

## DeepSeek V4 Flash 0731 local profiles

The bundled Strix and DGX files each provide two serving policies:

- the standard profile uses a 131,072-token context and retains the optional
  Q8-to-F16 acceleration cache;
- Strix `long-safe` uses a 240,000-token context and exports
  `DS4_CUDA_NO_Q8_F16_CACHE=1`. The retained ROCm engine automatically skips
  packed grouped-F16 attention under this cache policy because that path
  requires the optional F16 output-projection cache. The independent
  `DS4_ROCM_ATTN_DISABLE_PACKED_F16=1` rollback remains diagnostic-only;
- DGX `long-safe` is the single-session 300,000-token policy. It exports
  `DS4_CUDA_NO_Q8_F16_CACHE=1`, `DS4_SERVER_COALESCE=0`, and
  `DS4_METAL_PREFILL_CHUNK=2048`. Disabling server coalescing removes the second
  persistent graph and KV-bank pool; 2K prefill lowers the full serial graph's
  peak. This deliberately trades concurrent/continuous batching and speculative
  decode for enough UMA reserve to keep one long agent session resident.

Both policies expose `POWER` for all five executables and pass `--power`, with a
default GPU duty cycle of `100`. They cap responses at 30,000 tokens, keep
separate disk-KV directories so numerically different runtime policies cannot
reuse each other's checkpoint, and bind the server to `0.0.0.0:8000`. Disk KV
persists resumable prefixes; it does not move the active session out of RAM.

`profiles/deepseek-v4-flash-0731-strix.conf` binds `~/ds4-rocm-0731`, uses the
8K ROCm prefill chunk, relies on the engine's cache-aware attention gate for
long-safe, and deliberately omits `--backend rocm` from `ds4-agent`. Its profile
IDs are:

- `deepseek-v4-flash-0731-strix`
- `deepseek-v4-flash-0731-strix-long-safe`

`profiles/deepseek-v4-flash-0731-dgx.conf` binds the Entrpi CUDA build at
`~/code/ds4` and the exact model under `~/gguf`. Build that checkout with
`make cuda-spark`; `make cuda CUDA_ARCH=sm_121` omits the Spark HBM-cache build
policy. The fast profile leaves prefill chunking at the build default. The
session-safe profile uses the backend environment because `--prefill-chunk` is
not accepted consistently across all five executables. Its profile IDs are:

- `deepseek-v4-flash-0731-dgx`
- `deepseek-v4-flash-0731-dgx-long-safe`

Install the hardware-specific file as the user profile:

```sh
mkdir -p ~/.config/ds4-launcher
install -m 0644 profiles/deepseek-v4-flash-0731-strix.conf \
  ~/.config/ds4-launcher/profiles.conf       # Strix Halo
install -m 0644 profiles/deepseek-v4-flash-0731-dgx.conf \
  ~/.config/ds4-launcher/profiles.conf       # DGX Spark
```

Always inspect the complete environment and command without starting DS4:

```sh
ds4-launcher ds4-server --profile deepseek-v4-flash-0731-strix-long-safe \
  --no-prompt --dry-run
ds4-launcher ds4-server --profile deepseek-v4-flash-0731-dgx-long-safe \
  --no-prompt --dry-run
```

## DGX Spark and Strix Halo

The launcher is hardware-neutral. Build each bound checkout with its exact
platform target; a generic CUDA build is not equivalent to the Spark policy:

```sh
# DGX Spark profiles bind this checkout.
cd ~/code/ds4 && make cuda-spark -j"$(nproc)"

# Strix Halo profiles bind this checkout.
cd ~/ds4-rocm-0731 && make strix-halo -j"$(nproc)"
```

Use `--dry-run` after changing DS4 branches because supported flags may evolve.
The launcher intentionally prints the complete command and environment before
execution.
