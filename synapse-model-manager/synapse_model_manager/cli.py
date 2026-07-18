from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .core import (
    JobConflictError,
    JobStore,
    Manifest,
    ManifestError,
    ModelManagerError,
    SourceNotFoundError,
    ValidationError,
    discover_roots,
    enqueue_job,
    find_complete_source,
    install_manifest,
    list_jobs,
    load_registry,
    load_state,
    pending_jobs,
    remove_job,
    request_job_control,
    retry_job,
    utc_now,
    validate_manifest,
)

SCHEMA_VERSION = "1"


class Output:
    def __init__(self, json_mode: bool, jsonl_mode: bool) -> None:
        self.json_mode = json_mode
        self.jsonl_mode = jsonl_mode

    def event(self, event: str, details: dict[str, Any]) -> None:
        payload = {"schemaVersion": SCHEMA_VERSION, "event": event, "timestamp": utc_now(), **details}
        if self.jsonl_mode:
            print(json.dumps(payload, sort_keys=True), flush=True)
        elif not self.json_mode:
            summary = details.get("path") or details.get("component") or details.get("model") or ""
            print(f"[{event}] {summary}".rstrip(), file=sys.stderr, flush=True)

    def final(self, command: str, ok: bool, data: Any, warnings: list[Any], errors: list[Any]) -> None:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "command": command,
            "ok": ok,
            "timestamp": utc_now(),
            "data": data,
            "warnings": warnings,
            "errors": errors,
        }
        if self.json_mode:
            print(json.dumps(payload, sort_keys=True))
        elif self.jsonl_mode:
            print(json.dumps({"schemaVersion": SCHEMA_VERSION, "event": "result", **payload}, sort_keys=True))
        else:
            if ok:
                print(json.dumps(data, indent=2, sort_keys=True))
            else:
                for error in errors:
                    print(f"error: {error.get('message', error)}", file=sys.stderr)


def default_manifest_dirs(extra: list[str]) -> list[Path]:
    result = [Path(value) for value in extra]
    configured = os.environ.get("SYNAPSE_MODEL_MANIFEST_DIRS", "")
    result.extend(Path(value) for value in configured.split(":") if value)
    result.append(Path("/usr/share/synapse/models.d"))
    return result


def _extract_output_flags(argv: list[str]) -> tuple[list[str], bool, bool]:
    json_mode = "--json" in argv
    jsonl_mode = "--jsonl" in argv
    if json_mode and jsonl_mode:
        raise ManifestError("--json and --jsonl are mutually exclusive")
    return [value for value in argv if value not in {"--json", "--jsonl"}], json_mode, jsonl_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse-model", description="Manage external Synapse Linux AI models")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--manifest-dir", action="append", default=[], help="Additional model manifest directory")
    parser.add_argument("--state-dir", default=os.environ.get("SYNAPSE_MODEL_STATE_DIR", "/var/lib/synapse/model-manager"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered model bundles")
    show = subparsers.add_parser("show", help="Show a model manifest")
    show.add_argument("model")
    subparsers.add_parser("discover", help="Discover local model archives")

    status = subparsers.add_parser("status", help="Show activation and validation status")
    status.add_argument("model", nargs="?")
    status.add_argument("--root")

    verify = subparsers.add_parser("verify", help="Fully verify a model bundle")
    verify.add_argument("model")
    verify.add_argument("--source", default="auto", help="auto, usb, or an archive root path")
    verify.add_argument("--skip-hash", action="store_true")

    install = subparsers.add_parser("install", help="Acquire and activate a model bundle")
    install.add_argument("model")
    install.add_argument("--source", default="auto", help="auto, usb, web, or an archive root path")
    install.add_argument("--mode", choices=("external", "copy"), default="copy")
    install.add_argument("--root", help="Destination model root for copy mode")
    install.add_argument("--skip-hash", action="store_true")
    install.add_argument("--restart", action="store_true", help="Discard persistent partial data and restart")

    enqueue = subparsers.add_parser("enqueue", help="Add a persistent transfer to the background queue")
    enqueue.add_argument("model")
    enqueue.add_argument("--source", default="auto", help="auto, usb, web, or an archive root path")
    enqueue.add_argument("--mode", choices=("external", "copy"), default="copy")
    enqueue.add_argument("--root", help="Destination model root for copy mode")
    enqueue.add_argument("--skip-hash", action="store_true")
    enqueue.add_argument("--restart", action="store_true", help="Replace incompatible queued state and discard partials when run")
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--no-start-worker", action="store_true", help="Do not ask systemd to start the system worker")

    jobs = subparsers.add_parser("jobs", help="List persistent transfer jobs")
    jobs.add_argument("model", nargs="?")
    pause = subparsers.add_parser("pause", help="Request cooperative transfer pause")
    pause.add_argument("model")
    resume = subparsers.add_parser("resume", help="Resume a persistent transfer job")
    resume.add_argument("model")
    cancel = subparsers.add_parser("cancel", help="Request cooperative transfer cancellation")
    cancel.add_argument("model")
    retry = subparsers.add_parser("retry", help="Return a stopped or failed transfer to the queue")
    retry.add_argument("model")
    remove = subparsers.add_parser("remove-job", help="Remove a stopped queue job")
    remove.add_argument("model")
    remove.add_argument("--partials", action="store_true", help="Also remove its persistent staging directory")
    worker = subparsers.add_parser("worker", help="Process the persistent transfer queue")
    worker.add_argument("--wait", action="store_true", help="Keep polling after the queue becomes empty")
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--max-jobs", type=int, default=0, help="Maximum attempts before exit; zero is unlimited")

    resolve = subparsers.add_parser("resolve", help="Resolve activated component paths")
    resolve.add_argument("model")
    return parser


def require_manifest(registry: dict[str, Manifest], model_id: str) -> Manifest:
    try:
        return registry[model_id]
    except KeyError as exc:
        raise ManifestError(f"unknown model bundle: {model_id}") from exc


def _source_configuration(value: str) -> tuple[str, Path | None]:
    if value in {"auto", "usb", "web"}:
        return value, None
    return "usb", Path(value)


def _start_system_worker(state_dir: Path) -> bool:
    if state_dir != Path("/var/lib/synapse/model-manager") or os.geteuid() != 0 or shutil.which("systemctl") is None:
        return False
    result = subprocess.run(
        ["systemctl", "start", "--no-block", "synapse-model-worker.service"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _process_queue(
    registry: dict[str, Manifest],
    state_dir: Path,
    output: Output,
    *,
    wait: bool,
    poll_seconds: float,
    max_jobs: int,
) -> dict[str, Any]:
    if poll_seconds < 0.1:
        raise ManifestError("--poll-seconds must be at least 0.1")
    attempts: list[dict[str, Any]] = []
    while True:
        candidates = pending_jobs(state_dir)
        if not candidates:
            if wait:
                time.sleep(poll_seconds)
                continue
            break
        made_attempt = False
        for job in candidates:
            model_id = str(job.get("model", ""))
            try:
                manifest = require_manifest(registry, model_id)
            except ModelManagerError as exc:
                store = JobStore(state_dir, model_id)
                store.acquire()
                try:
                    store.set_state("failed", error=str(exc))
                finally:
                    store.release()
                attempts.append({"model": model_id, "ok": False, "error": {"code": exc.code, "message": str(exc)}})
                made_attempt = True
                continue
            explicit_value = job.get("explicitSource")
            output.event("worker-started-job", {"model": model_id, "state": job.get("state")})
            try:
                result = install_manifest(
                    manifest,
                    source=str(job.get("source", "auto")),
                    mode=str(job.get("mode", "copy")),
                    destination_root=Path(str(job.get("destinationRoot", manifest.default_root))),
                    state_dir=state_dir,
                    explicit_source=Path(str(explicit_value)) if explicit_value else None,
                    verify_hashes=bool(job.get("verifyHashes", True)),
                    emitter=output.event,
                    restart=bool(job.get("restartRequested", False)),
                )
                attempts.append({"model": model_id, "ok": True, "result": result})
                made_attempt = True
            except JobConflictError:
                output.event("worker-skipped-active-job", {"model": model_id})
                continue
            except ModelManagerError as exc:
                attempts.append({"model": model_id, "ok": False, "error": {"code": exc.code, "message": str(exc)}})
                made_attempt = True
            if max_jobs and len(attempts) >= max_jobs:
                return {"attempts": attempts, "pending": len(pending_jobs(state_dir))}
        if not wait:
            break
        if not made_attempt:
            time.sleep(poll_seconds)
    return {"attempts": attempts, "pending": len(pending_jobs(state_dir))}


def manifest_dict(manifest: Manifest) -> dict[str, Any]:
    return {
        "id": manifest.id,
        "name": manifest.name,
        "description": manifest.description,
        "product": manifest.product,
        "defaultRoot": manifest.default_root,
        "license": manifest.license,
        "manifest": str(manifest.path),
        "components": [
            {
                "id": component.id,
                "relativePath": component.relative_path,
                "sourceType": component.source_type,
                "repository": component.repo,
                "revision": component.revision,
                "requiredFiles": len(component.required),
                "requiredGlobs": list(component.required_globs),
            }
            for component in manifest.components
        ],
    }


def run_command(args: argparse.Namespace, registry: dict[str, Manifest], output: Output) -> Any:
    state_dir = Path(args.state_dir)
    if args.command == "list":
        return {"models": [manifest_dict(registry[key]) for key in sorted(registry)]}
    if args.command == "show":
        return manifest_dict(require_manifest(registry, args.model))
    if args.command == "discover":
        roots = discover_roots()
        values = []
        for root in roots:
            available = []
            for manifest in registry.values():
                report = validate_manifest(manifest, root, verify_hashes=False)
                if report.valid:
                    available.append(manifest.id)
            values.append({"root": str(root), "models": sorted(available)})
        return {"sources": values}
    if args.command == "status":
        state = load_state(state_dir)
        ids = [args.model] if args.model else sorted(registry)
        values = []
        for model_id in ids:
            manifest = require_manifest(registry, model_id)
            activation = state["models"].get(model_id)
            root = Path(args.root or (activation or {}).get("root", manifest.default_root))
            report = validate_manifest(manifest, root, verify_hashes=False)
            values.append({
                "model": model_id,
                "active": activation is not None,
                "mode": (activation or {}).get("mode"),
                "root": str(root),
                "validation": report.as_dict(),
            })
        return {"models": values}
    if args.command == "verify":
        manifest = require_manifest(registry, args.model)
        explicit = None if args.source in {"auto", "usb"} else Path(args.source)
        roots = discover_roots([explicit] if explicit else None)
        root, _ = find_complete_source(manifest, roots, verify_hashes=False)
        report = validate_manifest(
            manifest, root, verify_hashes=not args.skip_hash, emitter=output.event,
        )
        if not report.valid:
            raise ValidationError(f"model bundle {manifest.id} failed verification at {root}: {report.as_dict()}")
        return {"model": manifest.id, "validation": report.as_dict()}
    if args.command == "install":
        manifest = require_manifest(registry, args.model)
        source_kind, explicit = _source_configuration(args.source)
        destination = Path(args.root or manifest.default_root)
        return install_manifest(
            manifest,
            source=source_kind,
            mode=args.mode,
            destination_root=destination,
            state_dir=state_dir,
            explicit_source=explicit,
            verify_hashes=not args.skip_hash,
            emitter=output.event,
            restart=args.restart,
        )
    if args.command == "enqueue":
        manifest = require_manifest(registry, args.model)
        source_kind, explicit = _source_configuration(args.source)
        value = enqueue_job(
            state_dir,
            manifest.id,
            source=source_kind,
            mode=args.mode,
            destination_root=Path(args.root or manifest.default_root),
            explicit_source=explicit,
            verify_hashes=not args.skip_hash,
            restart=args.restart,
            priority=args.priority,
        )
        worker_started = False if args.no_start_worker else _start_system_worker(state_dir)
        return {"job": value, "workerStarted": worker_started}
    if args.command == "jobs":
        return {"jobs": list_jobs(state_dir, args.model)}
    if args.command in {"pause", "cancel"}:
        value = request_job_control(state_dir, args.model, args.command)
        return {"job": value}
    if args.command == "retry":
        return {"job": retry_job(state_dir, args.model)}
    if args.command == "remove-job":
        return remove_job(state_dir, args.model, partials=args.partials)
    if args.command == "worker":
        return _process_queue(
            registry,
            state_dir,
            output,
            wait=args.wait,
            poll_seconds=args.poll_seconds,
            max_jobs=args.max_jobs,
        )
    if args.command == "resume":
        manifest = require_manifest(registry, args.model)
        jobs = list_jobs(state_dir, args.model)
        if not jobs:
            raise SourceNotFoundError(f"no transfer job exists for {args.model}")
        job = jobs[0]
        explicit_value = job.get("explicitSource")
        return install_manifest(
            manifest,
            source=str(job.get("source", "auto")),
            mode=str(job.get("mode", "copy")),
            destination_root=Path(str(job.get("destinationRoot", manifest.default_root))),
            state_dir=state_dir,
            explicit_source=Path(explicit_value) if explicit_value else None,
            verify_hashes=bool(job.get("verifyHashes", True)),
            emitter=output.event,
        )
    if args.command == "resolve":
        manifest = require_manifest(registry, args.model)
        activation = load_state(state_dir)["models"].get(manifest.id)
        if not activation:
            raise SourceNotFoundError(f"model bundle {manifest.id} is not activated")
        root = Path(activation["root"])
        report = validate_manifest(manifest, root, verify_hashes=False)
        if not report.valid:
            raise SourceNotFoundError(f"activated model bundle {manifest.id} is incomplete")
        return {
            "model": manifest.id,
            "mode": activation["mode"],
            "root": str(root),
            "components": [
                {"id": component.id, "path": str(root / component.relative_path)}
                for component in manifest.components
            ],
        }
    raise ManifestError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    source_argv = list(sys.argv[1:] if argv is None else argv)
    command = source_argv[0] if source_argv else "unknown"
    try:
        clean_argv, json_mode, jsonl_mode = _extract_output_flags(source_argv)
    except ModelManagerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    output = Output(json_mode, jsonl_mode)
    parser = build_parser()
    try:
        args = parser.parse_args(clean_argv)
        command = args.command
        registry = load_registry(default_manifest_dirs(args.manifest_dir))
        if not registry:
            raise ManifestError("no model manifests were found")
        data = run_command(args, registry, output)
        output.final(command, True, data, [], [])
        return 0
    except ModelManagerError as exc:
        error = {"code": exc.code, "message": str(exc)}
        output.final(command, False, None, [], [error])
        return exc.exit_code
    except KeyboardInterrupt:
        error = {"code": "interrupted", "message": "operation interrupted"}
        output.final(command, False, None, [], [error])
        return 130
    except Exception as exc:  # Fail closed while preserving a machine-readable error.
        error = {"code": "internal_error", "message": str(exc), "type": type(exc).__name__}
        output.final(command, False, None, [], [error])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
