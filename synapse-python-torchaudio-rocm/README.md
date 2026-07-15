# synapse-python-torchaudio-rocm

System TorchAudio runtime for Synapse Linux ROCm applications.

The package installs the official PyTorch ROCm 7.1 CPython 3.14 wheel and uses
the distribution-provided `python-pytorch-opt-rocm` tensor runtime. It exists as
a separate package so applications do not vendor or download duplicate compiled
TorchAudio runtimes.

Validated on AMD Strix Halo with system PyTorch 2.12.1/ROCm 7.2 using:

- `import torchaudio`;
- GPU `torchaudio.functional.resample`;
- Coqui XTTS-v2;
- pyannote.audio;
- Chatterbox TTS.
