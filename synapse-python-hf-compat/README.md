# Synapse Python Hugging Face compatibility runtime

Pacman-owned shared runtime for version-sensitive Hugging Face components. It
contains Transformers, Tokenizers, SentencePiece, Hugging Face Hub, and
Accelerate once for all Synapse applications. Applications prepend
`/usr/lib/synapse-python-hf-compat` to `PYTHONPATH` and continue using system
PyTorch. No venv and no duplicate PyTorch are created.
