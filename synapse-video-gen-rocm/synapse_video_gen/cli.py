from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
DEFAULT_CONFIG = Path("/etc/synapse/video-gen.conf")
DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, static, blurred details, subtitles, watermark, worst quality, "
    "low quality, JPEG artifacts, deformed, disfigured, duplicated limbs"
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("\"'")
    return values


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), file=sys.stderr)
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def probe_duration(video: Path) -> float:
    result = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(video),
        ],
        capture=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source video has no positive finite duration")
    return duration


def reference_frame(video: Path, position: float, width: int, height: int):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(position, 0.0) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode source frame at {position:.3f}s")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        scale = max(width / image.width, height / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)))
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        return resized.crop((left, top, left + width, top + height))
    finally:
        capture.release()


def valid_frame_count(seconds: float, fps: int) -> int:
    requested = max(9, round(seconds * fps))
    return max(9, ((requested - 1 + 3) // 4) * 4 + 1)


def validate_model(model_root: Path) -> list[str]:
    required = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/spiece.model",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    missing = [relative for relative in required if not (model_root / relative).is_file()]
    if not any((model_root / "transformer").glob("*.safetensors")):
        missing.append("transformer/*.safetensors")
    if not any((model_root / "text_encoder").glob("*.safetensors")):
        missing.append("text_encoder/*.safetensors")
    return missing


def load_pipeline(model_root: Path):
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm does not expose the AMD GPU through torch.cuda")
    vae = AutoencoderKLWan.from_pretrained(
        model_root, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipeline = WanVACEPipeline.from_pretrained(
        model_root, vae=vae, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config, flow_shift=3.0
    )
    pipeline.to("cuda")
    return pipeline


def concatenate_chunks(chunks: list[Path], output: Path, duration: float, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        concat_path = Path(handle.name)
        for chunk in chunks:
            escaped = str(chunk.resolve()).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
    try:
        run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", str(concat_path), "-t", f"{duration:.6f}", "-an",
                "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ]
        )
    finally:
        concat_path.unlink(missing_ok=True)


def generate(args: argparse.Namespace) -> int:
    import torch
    from diffusers.utils import export_to_video

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Source video does not exist: {source}")
    missing = validate_model(model_root)
    if missing:
        raise RuntimeError("Incomplete Wan VACE model: " + ", ".join(missing))
    if args.width % 16 or args.height % 16:
        raise ValueError("Width and height must be divisible by 16")
    duration = probe_duration(source)
    if duration > args.max_duration:
        raise ValueError(
            f"Source duration {duration:.2f}s exceeds --max-duration {args.max_duration:.2f}s"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = output.parent / f".{output.stem}-chunks"
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    chunk_count = max(1, math.ceil(duration / args.chunk_seconds))
    pipeline = load_pipeline(model_root)
    chunks: list[Path] = []
    started = time.time()
    for index in range(chunk_count):
        start = index * args.chunk_seconds
        seconds = min(args.chunk_seconds, duration - start)
        frames = valid_frame_count(seconds, args.fps)
        reference = reference_frame(source, start, args.width, args.height)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        print(f"Generating shot {index + 1}/{chunk_count}: {frames} frames", file=sys.stderr)
        result = pipeline(
            reference_images=[reference],
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).frames[0]
        chunk = chunk_dir / f"shot-{index:04d}.mp4"
        export_to_video(result, str(chunk), fps=args.fps)
        chunks.append(chunk)
    concatenate_chunks(chunks, output, duration, args.fps)
    manifest = {
        "source": str(source),
        "output": str(output),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "model_root": str(model_root),
        "model_revision": MODEL_REVISION,
        "duration": duration,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "shots": len(chunks),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output.with_suffix(".generation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.keep_chunks:
        shutil.rmtree(chunk_dir)
    print(output)
    return 0


def doctor(args: argparse.Namespace) -> int:
    config = load_config()
    model_root = args.model_root or Path(config.get("MODEL_ROOT", "/var/lib/synapse/video-gen/models/wan-vace-1.3b"))
    checks: list[tuple[str, bool, str]] = []
    for command in ("ffmpeg", "ffprobe"):
        path = shutil.which(command)
        checks.append((command, bool(path), path or "missing"))
    try:
        import torch
        checks.append(("PyTorch ROCm", torch.cuda.is_available(), torch.__version__))
    except Exception as exc:
        checks.append(("PyTorch ROCm", False, str(exc)))
    for module in ("diffusers", "transformers", "tokenizers", "sentencepiece", "accelerate"):
        try:
            imported = __import__(module)
            checks.append((module, True, getattr(imported, "__version__", "available")))
        except Exception as exc:
            checks.append((module, False, str(exc)))
    missing = validate_model(model_root)
    checks.append(("Wan VACE model", not missing, str(model_root) if not missing else ", ".join(missing)))
    failed = False
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name:20} {detail}")
        failed |= not ok
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    config = load_config()
    default_root = Path(config.get("MODEL_ROOT", "/var/lib/synapse/video-gen/models/wan-vace-1.3b"))
    parser = argparse.ArgumentParser(prog="synapse-video-gen", description="ROCm generative video backend")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--model-root", type=Path, default=default_root)
    doctor_parser.set_defaults(handler=doctor)
    generate_parser = commands.add_parser("generate", help="Generate new visual shots from a real-video reference")
    generate_parser.add_argument("--input", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--model-root", type=Path, default=default_root)
    generate_parser.add_argument("--width", type=int, default=832)
    generate_parser.add_argument("--height", type=int, default=480)
    generate_parser.add_argument("--fps", type=int, default=16)
    generate_parser.add_argument("--steps", type=int, default=30)
    generate_parser.add_argument("--guidance-scale", type=float, default=5.0)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--chunk-seconds", type=float, default=5.0)
    generate_parser.add_argument("--max-duration", type=float, default=60.0)
    generate_parser.add_argument("--keep-chunks", action="store_true")
    generate_parser.set_defaults(handler=generate)
    return parser


def main() -> int:
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    try:
        args = build_parser().parse_args()
        return int(args.handler(args))
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"synapse-video-gen: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
