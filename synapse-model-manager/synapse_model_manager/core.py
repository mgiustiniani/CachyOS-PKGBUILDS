from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python >= 3.11 is required by the package
    tomllib = None


class ModelManagerError(RuntimeError):
    exit_code = 1
    code = "model_manager_error"


class ManifestError(ModelManagerError):
    exit_code = 2
    code = "manifest_error"


class SourceNotFoundError(ModelManagerError):
    exit_code = 3
    code = "source_not_found"


class ValidationError(ModelManagerError):
    exit_code = 4
    code = "validation_failed"


class AcquisitionError(ModelManagerError):
    exit_code = 5
    code = "acquisition_failed"


@dataclass(frozen=True)
class RequiredFile:
    path: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class HttpFile:
    path: str
    url: str
    size: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Component:
    id: str
    relative_path: str
    source_type: str
    repo: str | None = None
    revision: str | None = None
    revision_required: bool = True
    include: tuple[str, ...] = ()
    required: tuple[RequiredFile, ...] = ()
    required_globs: tuple[str, ...] = ()
    http_files: tuple[HttpFile, ...] = ()


@dataclass(frozen=True)
class Manifest:
    id: str
    name: str
    description: str
    product: str
    default_root: str
    license: str
    components: tuple[Component, ...]
    path: Path


@dataclass
class ValidationReport:
    valid: bool
    root: Path
    checked_files: int = 0
    checked_bytes: int = 0
    missing: list[str] = field(default_factory=list)
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "root": str(self.root),
            "checkedFiles": self.checked_files,
            "checkedBytes": self.checked_bytes,
            "missing": self.missing,
            "mismatches": self.mismatches,
            "revisions": self.revisions,
        }


Emitter = Callable[[str, dict[str, Any]], None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_relative(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in ("", "."):
        raise ManifestError(f"{label} must be a non-empty safe relative path: {value!r}")
    return path.as_posix()


def load_manifest(path: Path) -> Manifest:
    if tomllib is None:
        raise ManifestError("Python tomllib is unavailable")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise ManifestError(f"unsupported schema_version in {path}")
    components: list[Component] = []
    for item in raw.get("components", []):
        relative_path = _safe_relative(str(item.get("relative_path", "")), "component.relative_path")
        required = tuple(
            RequiredFile(
                path=_safe_relative(str(entry.get("path", "")), "required.path"),
                size=int(entry["size"]) if "size" in entry else None,
                sha256=str(entry["sha256"]).lower() if entry.get("sha256") else None,
            )
            for entry in item.get("required", [])
        )
        http_files = tuple(
            HttpFile(
                path=_safe_relative(str(entry.get("path", "")), "http_files.path"),
                url=str(entry.get("url", "")),
                size=int(entry["size"]) if "size" in entry else None,
                sha256=str(entry["sha256"]).lower() if entry.get("sha256") else None,
            )
            for entry in item.get("http_files", [])
        )
        source_type = str(item.get("source_type", "huggingface"))
        if source_type not in {"huggingface", "http-files", "external-only"}:
            raise ManifestError(f"unsupported source_type {source_type!r} in {path}")
        if source_type == "huggingface" and not item.get("repo"):
            raise ManifestError(f"component {item.get('id')} has no Hugging Face repo")
        components.append(
            Component(
                id=str(item.get("id", "")),
                relative_path=relative_path,
                source_type=source_type,
                repo=str(item["repo"]) if item.get("repo") else None,
                revision=str(item["revision"]) if item.get("revision") else None,
                revision_required=bool(item.get("revision_required", True)),
                include=tuple(str(value) for value in item.get("include", [])),
                required=required,
                required_globs=tuple(str(value) for value in item.get("required_globs", [])),
                http_files=http_files,
            )
        )
    if not components:
        raise ManifestError(f"manifest {path} has no components")
    return Manifest(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", raw.get("id", ""))),
        description=str(raw.get("description", "")),
        product=str(raw.get("product", raw.get("id", ""))),
        default_root=str(raw.get("default_root", "/var/lib/synapse/models")),
        license=str(raw.get("license", "unknown")),
        components=tuple(components),
        path=path,
    )


def load_registry(manifest_dirs: list[Path]) -> dict[str, Manifest]:
    manifests: dict[str, Manifest] = {}
    for directory in manifest_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            manifest = load_manifest(path)
            if not manifest.id:
                raise ManifestError(f"manifest {path} has no id")
            if manifest.id in manifests:
                raise ManifestError(f"duplicate model id {manifest.id!r}")
            manifests[manifest.id] = manifest
    return manifests


def sha256_file(path: Path, emitter: Emitter | None = None) -> str:
    digest = hashlib.sha256()
    completed = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            completed += len(chunk)
            if emitter and completed % (512 * 1024 * 1024) < len(chunk):
                emitter("hash-progress", {"path": str(path), "completedBytes": completed})
    return digest.hexdigest()


def _sidecar_hashes(component_root: Path) -> dict[str, str]:
    sidecar = component_root / "SHA256SUMS.synapse"
    values: dict[str, str] = {}
    if not sidecar.is_file():
        return values
    for line in sidecar.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        relative = parts[1].lstrip("* ")
        try:
            relative = _safe_relative(relative, "SHA256SUMS entry")
        except ManifestError:
            continue
        values[relative] = parts[0].lower()
    return values


def validate_manifest(
    manifest: Manifest,
    root: Path,
    *,
    verify_hashes: bool,
    emitter: Emitter | None = None,
) -> ValidationReport:
    report = ValidationReport(valid=True, root=root)
    for component in manifest.components:
        component_root = root / component.relative_path
        sidecar = _sidecar_hashes(component_root) if verify_hashes else {}
        if component.revision:
            revision_path = component_root / ".snapshot-revision"
            actual = revision_path.read_text(encoding="utf-8").strip() if revision_path.is_file() else None
            revision_ok = actual == component.revision or (actual is None and not component.revision_required)
            report.revisions.append({
                "component": component.id,
                "expected": component.revision,
                "actual": actual,
                "required": component.revision_required,
                "valid": revision_ok,
            })
            if not revision_ok:
                report.mismatches.append({
                    "path": str(revision_path),
                    "kind": "revision",
                    "expected": component.revision,
                    "actual": actual,
                })
        for required in component.required:
            path = component_root / required.path
            if not path.is_file():
                report.missing.append(str(path))
                continue
            size = path.stat().st_size
            report.checked_files += 1
            report.checked_bytes += size
            if required.size is not None and size != required.size:
                report.mismatches.append({
                    "path": str(path), "kind": "size", "expected": required.size, "actual": size,
                })
            expected_hash = required.sha256 or sidecar.get(required.path)
            if verify_hashes and expected_hash:
                if emitter:
                    emitter("verifying", {"component": component.id, "path": str(path), "sizeBytes": size})
                actual_hash = sha256_file(path, emitter)
                if actual_hash != expected_hash:
                    report.mismatches.append({
                        "path": str(path), "kind": "sha256", "expected": expected_hash, "actual": actual_hash,
                    })
        for pattern in component.required_globs:
            matches = [path for path in component_root.glob(pattern) if path.is_file()]
            if not matches:
                report.missing.append(str(component_root / pattern))
            else:
                report.checked_files += len(matches)
                report.checked_bytes += sum(path.stat().st_size for path in matches)
        if not component.required and not component.required_globs and not component_root.exists():
            report.missing.append(str(component_root))
    report.valid = not report.missing and not report.mismatches
    return report


def _findmnt_label_models() -> list[Path]:
    try:
        result = subprocess.run(
            ["findmnt", "-rn", "-S", "LABEL=models", "-o", "TARGET"],
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return []
    return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def discover_roots(extra: list[Path] | None = None) -> list[Path]:
    candidates: list[Path] = []
    if extra:
        candidates.extend(extra)
    env_root = os.environ.get("SYNAPSE_MODEL_SOURCE")
    if env_root:
        candidates.append(Path(env_root))
    mounts = _findmnt_label_models()
    candidates.extend(mounts)
    candidates.extend([Path("/mnt/models"), Path("/media/models")])
    for pattern in ("/run/media/*/models", "/run/media/*/*/models"):
        candidates.extend(Path("/").glob(pattern.lstrip("/")))
    expanded: list[Path] = []
    for candidate in candidates:
        expanded.extend([candidate / "models", candidate])
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in expanded:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        result.append(candidate)
    return result


def find_complete_source(
    manifest: Manifest,
    candidates: list[Path],
    *,
    verify_hashes: bool = False,
) -> tuple[Path, ValidationReport]:
    reports: list[ValidationReport] = []
    for candidate in candidates:
        report = validate_manifest(manifest, candidate, verify_hashes=verify_hashes)
        reports.append(report)
        if report.valid:
            return candidate, report
    detail = "; ".join(f"{report.root}: {len(report.missing)} missing, {len(report.mismatches)} mismatches" for report in reports)
    raise SourceNotFoundError(f"no complete source found for {manifest.id}" + (f" ({detail})" if detail else ""))


def _run_hf_download(component: Component, target: Path, emitter: Emitter) -> None:
    if shutil.which("hf") is None:
        raise AcquisitionError("the hf CLI from python-huggingface-hub is required")
    target.mkdir(parents=True, exist_ok=True)
    command = ["hf", "download", str(component.repo)]
    if component.revision:
        command.extend(["--revision", component.revision])
    for pattern in component.include:
        command.extend(["--include", pattern])
    command.extend(["--local-dir", str(target)])
    emitter("download-started", {"component": component.id, "repo": component.repo, "revision": component.revision})
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise AcquisitionError(f"Hugging Face download failed for {component.id}") from exc
    if component.revision:
        (target / ".snapshot-revision").write_text(component.revision + "\n", encoding="utf-8")


def _download_http_file(item: HttpFile, target: Path, emitter: Emitter) -> None:
    destination = target / item.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(item.url, headers={"User-Agent": "SynapseModelManager/0.1"})
    emitter("download-started", {"path": str(destination), "url": item.url})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    except OSError as exc:
        raise AcquisitionError(f"download failed for {item.url}: {exc}") from exc
    partial.replace(destination)


def acquire_web(manifest: Manifest, staging_root: Path, emitter: Emitter) -> None:
    for component in manifest.components:
        target = staging_root / component.relative_path
        if component.source_type == "huggingface":
            _run_hf_download(component, target, emitter)
        elif component.source_type == "http-files":
            for item in component.http_files:
                _download_http_file(item, target, emitter)
            if component.revision:
                (target / ".snapshot-revision").write_text(component.revision + "\n", encoding="utf-8")
        else:
            raise AcquisitionError(f"component {component.id} is available only from an external archive")


def _copy_component(source: Path, destination: Path, emitter: Emitter, component: Component) -> None:
    emitter("copy-started", {"component": component.id, "source": str(source), "destination": str(destination)})
    selected: set[Path] = {source / required.path for required in component.required}
    for pattern in component.required_globs:
        selected.update(path for path in source.glob(pattern) if path.is_file())
    for metadata in (source / ".snapshot-revision", source / "SHA256SUMS.synapse"):
        if metadata.is_file():
            selected.add(metadata)
    if selected:
        for path in sorted(selected):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=False)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    emitter("copy-completed", {"component": component.id, "destination": str(destination)})


def stage_from_source(manifest: Manifest, source_root: Path, staging_root: Path, emitter: Emitter) -> None:
    for component in manifest.components:
        _copy_component(
            source_root / component.relative_path,
            staging_root / component.relative_path,
            emitter,
            component,
        )


def activate_staging(manifest: Manifest, staging_root: Path, destination_root: Path, emitter: Emitter) -> None:
    backup_root = destination_root / ".synapse-model-backup" / f"{manifest.id}-{uuid.uuid4().hex}"
    moved: list[tuple[Path, Path | None]] = []
    try:
        for component in manifest.components:
            staged_component = staging_root / component.relative_path
            for staged in sorted(path for path in staged_component.rglob("*") if path.is_file()):
                relative_file = staged.relative_to(staging_root)
                target = destination_root / relative_file
                target.parent.mkdir(parents=True, exist_ok=True)
                backup: Path | None = None
                if target.exists():
                    backup = backup_root / relative_file
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    target.replace(backup)
                staged.replace(target)
                moved.append((target, backup))
            emitter("activated", {"component": component.id, "path": str(destination_root / component.relative_path)})
    except OSError as exc:
        for target, backup in reversed(moved):
            if target.exists():
                target.unlink()
            if backup and backup.exists():
                backup.replace(target)
        raise AcquisitionError(f"activation failed: {exc}") from exc
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def load_state(state_dir: Path) -> dict[str, Any]:
    path = state_dir / "activations.json"
    if not path.is_file():
        return {"schemaVersion": "1", "models": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManagerError(f"cannot read activation state: {exc}") from exc
    if not isinstance(value.get("models"), dict):
        raise ModelManagerError("invalid activation state")
    return value


def save_activation(state_dir: Path, manifest: Manifest, root: Path, mode: str) -> None:
    state = load_state(state_dir)
    state["models"][manifest.id] = {
        "mode": mode,
        "root": str(root.resolve()),
        "updatedAt": utc_now(),
        "components": [
            {"id": component.id, "path": str((root / component.relative_path).resolve())}
            for component in manifest.components
        ],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / "activations.json"
    fd, temporary_name = tempfile.mkstemp(prefix="activations.", suffix=".json", dir=state_dir)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(state, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def install_manifest(
    manifest: Manifest,
    *,
    source: str,
    mode: str,
    destination_root: Path,
    state_dir: Path,
    explicit_source: Path | None,
    verify_hashes: bool,
    emitter: Emitter,
) -> dict[str, Any]:
    local_candidates = discover_roots([explicit_source] if explicit_source else None)
    source_root: Path | None = None
    source_report: ValidationReport | None = None
    if source not in {"web"}:
        try:
            source_root, source_report = find_complete_source(
                manifest, local_candidates, verify_hashes=verify_hashes,
            )
        except SourceNotFoundError:
            if source != "auto":
                raise
    if source_root is not None and mode == "external":
        save_activation(state_dir, manifest, source_root, "external")
        emitter("completed", {"model": manifest.id, "mode": "external", "root": str(source_root)})
        return {
            "model": manifest.id,
            "mode": "external",
            "source": str(source_root),
            "root": str(source_root),
            "validation": source_report.as_dict() if source_report else None,
        }
    if source_root is None and source == "usb":
        raise SourceNotFoundError(f"no complete USB source found for {manifest.id}")
    if source_root is None and mode == "external":
        raise AcquisitionError("web acquisition requires --mode copy")

    destination_root.mkdir(parents=True, exist_ok=True)
    staging_parent = destination_root / ".synapse-model-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = staging_parent / f"{manifest.id}-{uuid.uuid4().hex}"
    staging_root.mkdir()
    try:
        if source_root is not None:
            stage_from_source(manifest, source_root, staging_root, emitter)
            acquisition_source = str(source_root)
        else:
            acquire_web(manifest, staging_root, emitter)
            acquisition_source = "web"
        report = validate_manifest(manifest, staging_root, verify_hashes=verify_hashes, emitter=emitter)
        if not report.valid:
            raise ValidationError(f"acquired model failed validation: {report.as_dict()}")
        activate_staging(manifest, staging_root, destination_root, emitter)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    save_activation(state_dir, manifest, destination_root, "copy")
    emitter("completed", {"model": manifest.id, "mode": "copy", "root": str(destination_root)})
    return {
        "model": manifest.id,
        "mode": "copy",
        "source": acquisition_source,
        "root": str(destination_root),
        "validation": report.as_dict(),
    }
