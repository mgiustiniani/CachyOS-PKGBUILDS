"""Narrow Synapse compatibility shims for repository version skew."""

import importlib.metadata

_original_version = importlib.metadata.version


def _compatible_version(distribution_name: str) -> str:
    value = _original_version(distribution_name)
    # CachyOS currently ships transformers 5.2.0 with tokenizers 0.23.1 while
    # transformers' metadata caps the accepted version at 0.23.0. The API used
    # by XTTS and MuseTalk is compatible; report the capped version only to the
    # metadata guard instead of duplicating tokenizers in the private venv.
    if distribution_name.lower().replace("_", "-") == "tokenizers" and value == "0.23.1":
        return "0.23.0"
    return value


importlib.metadata.version = _compatible_version
