# Synapse XDNA Runtime Integration

This package completes the open XRT platform layer for AMD Strix Point and
Strix Halo XDNA2 NPUs. It installs:

- the AMD/Xilinx VTD validation archive omitted by the Arch
  `xrt-plugin-amdxdna` package;
- the non-root memlock policy required by XRT;
- `synapse-xdna doctor` and `synapse-xdna proof`;
- `synapse-xdna-import`, a post-install importer for the account-gated AMD SDK.

## Verify real NPU execution

Open a new login session after installation, then run:

```bash
synapse-xdna doctor
synapse-xdna proof
synapse-xdna proof --json
# After importing AMD Ryzen AI Software:
synapse-xdna onnx-proof --json
```

`proof` runs AMD's INT8 GEMM validation workload while polling XRT's AIE
partition report. It succeeds only if the validation passes **and** an active
hardware context is observed. Merely finding `/dev/accel/accel0` is not treated
as proof of execution.

## Import AMD Ryzen AI Software after installation

AMD Ryzen AI Software 1.7.1 is account-gated and governed by AMD's EULA. Its
archive and Xilinx license are never included in this package, the pacman
repository, or the Synapse ISO. The importer discovers them on a separately
mounted USB drive and installs them only into the target system:

```bash
synapse-xdna-import discover
sudo synapse-xdna-import install --source auto --license auto --accept-amd-eula
```

The acceptance flag is mandatory. The installer copies the private license to
mode-0600 target storage, records only its path in system configuration, and
runs AMD's isolated VitisAI ONNX quick test before reporting success.
Calamares exposes the same flow as an optional checkbox with explicit EULA
acceptance.

`synapse-xdna doctor --json` reports `platform_ready` and `onnx_npu_ready`
separately. The existing ROCm GPU stack is independent and is not modified.

## Sources

The validation archive is fetched from AMD/Xilinx VTD at the commit pinned in
`PKGBUILD`. VTD's README declares `SPDX-License-Identifier: Apache-2.0`.
