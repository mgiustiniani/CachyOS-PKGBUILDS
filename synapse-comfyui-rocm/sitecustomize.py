"""Narrow compatibility shim for the current CachyOS Transformers stack."""
import importlib.metadata
_original_version = importlib.metadata.version

def _compatible_version(name: str) -> str:
    value = _original_version(name)
    if name.lower().replace("_", "-") == "tokenizers" and value == "0.23.1":
        return "0.23.0"
    return value

importlib.metadata.version = _compatible_version
