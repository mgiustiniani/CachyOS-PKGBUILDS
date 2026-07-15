from __future__ import annotations

import argparse
from pathlib import Path

from .engine import engine


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speech with Synapse Voice AI")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("-o", "--output", default="speech.wav")
    parser.add_argument("--voice", default="default")
    parser.add_argument("--language", default="it")
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg-weight", type=float, default=0.5)
    args = parser.parse_args()
    payload = engine.synthesize(
        args.text,
        voice=args.voice,
        language=args.language,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )
    output = Path(args.output)
    output.write_bytes(payload)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
