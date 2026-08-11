# Synapse ROCm Runtime

System-runtime policy package for Synapse Linux ROCm applications. It reuses the
CachyOS/Arch ROCm and optimized PyTorch packages; it does not vendor another
ROCm or Python environment.

The dependency floor keeps the ROCm 7.2.4 stack paired with the glibc ABI used
by its `libamd_comgr.so.3` build. On a rolling-release installation, perform a
full system upgrade rather than installing glibc in isolation:

```sh
sudo pacman -Syu
sudo reboot
synapse-rocm-doctor --pretty
```

The doctor checks glibc, `libamd_comgr`, `/dev/kfd`, the render node,
`rocminfo`, and the system ROCm PyTorch device. It only reports state and never
changes the host.

Strix Halo machines may additionally install `synapse-strixhalo-config` for the
project's unified-memory kernel tuning.
