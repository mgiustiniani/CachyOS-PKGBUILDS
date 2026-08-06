# synapse-ds4-launcher

Synapse Linux packaging for the external DS4 profile launcher.

The package name uses the mandatory `synapse-` prefix because the launcher is built from a Synapse-owned source snapshot. The installed command intentionally remains:

```text
/usr/bin/ds4-launcher
```

## Pinned source

The packaged C source and upstream tests are copied byte-for-byte from:

```text
project: ds4-launcher source repository
commit:  1aa36cd899ce0da0101fbb59ec2567fd97ffbdd6
license: MIT
```

The source repository currently has no remote. Updating the snapshot requires an explicit commit review, refreshed checksums, a version/pkgrel update, and the package tests.

## Package boundary

`synapse-ds4-launcher` contains only the dependency-free policy launcher, documentation, and a portable system profile. It does not contain DS4, model weights, ROCm, host addresses, API credentials, or user-home paths.

The DS4 engine is packaged separately as `synapse-ds4-git`. This keeps the stable launcher policy independent from a moving architecture-specific engine build.

Installed profile locations:

```text
/etc/ds4-launcher/profiles.conf
/usr/share/synapse-ds4-launcher/profiles/synapse-ds4-server.conf
```

`/etc/ds4-launcher/profiles.conf` is a pacman backup file and is never silently overwritten on upgrade.

## Model paths

The system profile defaults to the canonical `synapse-model-manager` DS4 Q2 component path under:

```text
/var/lib/synapse/models/gguf/deepseek4-flash-284B-GGUF
```

Calamares selects the exact verified bundle and writes `/etc/ds4/config`. The `synapse-ds4-server` wrapper supplied by the engine package converts that root-owned configuration into explicit `--set` values and invokes `ds4-launcher` without shell-generated DS4 arguments.

## Validation

Inspect the complete command without starting DS4:

```sh
sudo -u root /usr/bin/synapse-ds4-server --dry-run
```

The package build runs the upstream CLI suite and a package-specific system-profile test.
