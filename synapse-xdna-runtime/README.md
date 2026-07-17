# Synapse XDNA Runtime Integration

This package completes the open XRT platform layer for AMD Strix Point and
Strix Halo XDNA2 NPUs. It installs:

- the AMD/Xilinx VTD validation archive omitted by the Arch
  `xrt-plugin-amdxdna` package;
- the non-root memlock policy required by XRT;
- `synapse-xdna doctor` and `synapse-xdna proof`.

## Verify real NPU execution

Open a new login session after installation, then run:

```bash
synapse-xdna doctor
synapse-xdna proof
synapse-xdna proof --json
```

`proof` runs AMD's INT8 GEMM validation workload while polling XRT's AIE
partition report. It succeeds only if the validation passes **and** an active
hardware context is observed. Merely finding `/dev/accel/accel0` is not treated
as proof of execution.

## ONNX Runtime boundary

This package does not claim to provide an ONNX NPU execution provider. Generic
ONNX inference additionally requires AMD Ryzen AI Software for Linux, including
`VitisAIExecutionProvider`, VOE, and the compiler/runtime libraries. AMD Ryzen
AI Software 1.7.1 is account-gated and governed by AMD's EULA, so its binaries
are not redistributed in this package or the Synapse repository.

`synapse-xdna doctor --json` reports `platform_ready` and `onnx_npu_ready`
separately. The existing ROCm GPU stack is independent and is not modified.

## Sources

The validation archive is fetched from AMD/Xilinx VTD at the commit pinned in
`PKGBUILD`. VTD's README declares `SPDX-License-Identifier: Apache-2.0`.
