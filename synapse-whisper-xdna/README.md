# Synapse Whisper XDNA

Runs AMD's static-shape Whisper large-v3-turbo ONNX encoder and decoder through
the Ryzen AI VitisAI execution provider on XDNA2. Model weights and tokenizer
assets are resolved through `synapse-model`; they are not included in this
package or the ISO.

```bash
synapse-whisper-xdna doctor
synapse-whisper-xdna warmup
synapse-whisper-xdna transcribe recording.wav --language it
synapse-whisper-xdna transcribe recording.wav --language it --json
synapse-whisper-xdna transcribe recording.wav --language it --timestamps --json
```

Audio is normalized to mono 16 kHz PCM and processed in 30-second chunks. The
optional timestamp mode returns speech segments suitable for downstream dubbing
and diarization workflows. The
encoder and decoder transformer run on VitisAI. With the current AMD graph, the
decoder's small token-embedding `Gather` remains on CPU; the computationally
heavy decoder partition executes on XDNA2.

The first `warmup` compiles model-specific AIE caches and can temporarily use
substantial CPU and NPU resources. Later runs load those caches and should be
much faster. Install the private AMD Ryzen AI runtime with
`synapse-xdna-import` before using this command.
