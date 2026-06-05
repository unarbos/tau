from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MANIFEST_VERSION = 1
DEFAULT_ENTRYPOINT = "agent.py"
SIGNATURE_PREFIX_V2 = "tau-private-submission-v2"
MAX_BUNDLE_FILES = 64
MAX_PATH_DEPTH = 8
MAX_BUNDLE_BYTES = 5_000_000
ALLOWED_SUFFIXES = frozenset({".py"})
HARNESS_EXCLUDE_DIR_NAMES = frozenset({".git", "scripts", "__pycache__", ".venv", ".cursor", ".github"})

_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")


@dataclass(slots=True, frozen=True)
class BundleFileRecord:
    path: str
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SubmissionManifest:
    version: int = MANIFEST_VERSION
    entrypoint: str = DEFAULT_ENTRYPOINT
    bundle_sha256: str = ""
    files: list[BundleFileRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entrypoint": self.entrypoint,
            "bundle_sha256": self.bundle_sha256,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SubmissionManifest:
        raw_files = payload.get("files") or []
        files = [
            BundleFileRecord(
                path=str(item["path"]),
                sha256=str(item["sha256"]).lower(),
                size=int(item["size"]),
            )
            for item in raw_files
            if isinstance(item, dict) and item.get("path")
        ]
        return cls(
            version=int(payload.get("version", MANIFEST_VERSION)),
            entrypoint=str(payload.get("entrypoint") or DEFAULT_ENTRYPOINT),
            bundle_sha256=str(payload.get("bundle_sha256") or "").lower(),
            files=files,
        )


def canonical_bundle_sha256(files: dict[str, bytes]) -> str:
    """Deterministic hash over sorted path:content_sha256 lines."""
    lines = [
        f"{path}:{hashlib.sha256(content).hexdigest()}"
        for path, content in sorted(normalize_bundle_files(files).items())
    ]
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(*, files: dict[str, bytes], entrypoint: str = DEFAULT_ENTRYPOINT) -> SubmissionManifest:
    normalized = normalize_bundle_files(files)
    violations = validate_bundle_paths(normalized.keys())
    if violations:
        raise ValueError("; ".join(violations))
    if entrypoint not in normalized:
        raise ValueError(f"bundle is missing required entrypoint `{entrypoint}`")

    records = [
        BundleFileRecord(
            path=path,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
        for path, content in sorted(normalized.items())
    ]
    bundle_sha256 = canonical_bundle_sha256(normalized)
    return SubmissionManifest(
        entrypoint=entrypoint,
        bundle_sha256=bundle_sha256,
        files=records,
    )


def normalize_bundle_files(files: dict[str, bytes]) -> dict[str, bytes]:
    normalized: dict[str, bytes] = {}
    for raw_path, content in files.items():
        path = normalize_relative_path(raw_path)
        if path in normalized:
            raise ValueError(f"duplicate bundle path `{path}`")
        normalized[path] = content
    return normalized


def normalize_relative_path(raw_path: str) -> str:
    path = PurePosixPath(str(raw_path).replace("\\", "/")).as_posix().lstrip("./")
    if not path or path.startswith("/"):
        raise ValueError(f"invalid bundle path `{raw_path}`")
    parts = [part for part in path.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"bundle path must not contain `.` or `..` segments: `{raw_path}`")
    return "/".join(parts)


def validate_bundle_paths(paths: Iterable[str]) -> list[str]:
    violations: list[str] = []
    seen: set[str] = set()
    path_list = list(paths)
    if len(path_list) > MAX_BUNDLE_FILES:
        violations.append(f"bundle has {len(path_list)} files; maximum is {MAX_BUNDLE_FILES}.")
    for raw_path in path_list:
        try:
            path = normalize_relative_path(raw_path)
        except ValueError as exc:
            violations.append(str(exc))
            continue
        if path in seen:
            violations.append(f"duplicate bundle path `{path}`.")
            continue
        seen.add(path)
        if path.count("/") + 1 > MAX_PATH_DEPTH:
            violations.append(f"bundle path `{path}` exceeds max depth {MAX_PATH_DEPTH}.")
        if not _PATH_RE.fullmatch(path):
            violations.append(f"bundle path `{path}` contains disallowed characters.")
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            violations.append(f"bundle path `{path}` must end with one of {sorted(ALLOWED_SUFFIXES)}.")
    if DEFAULT_ENTRYPOINT not in seen:
        violations.append(f"bundle must include `{DEFAULT_ENTRYPOINT}`.")
    return violations


def bundle_signature_payload(*, hotkey: str, submission_id: str, bundle_sha256: str) -> bytes:
    return f"{SIGNATURE_PREFIX_V2}:{hotkey}:{submission_id}:{bundle_sha256.lower()}".encode("utf-8")


def read_manifest(path: Path) -> SubmissionManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return SubmissionManifest.from_dict(payload)


def write_bundle_directory(
    *,
    root: Path,
    submission_id: str,
    files: dict[str, bytes],
    entrypoint: str = DEFAULT_ENTRYPOINT,
) -> tuple[Path, SubmissionManifest]:
    manifest = build_manifest(files=files, entrypoint=entrypoint)
    target = root / submission_id
    files_dir = target / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for record in manifest.files:
        out_path = files_dir / record.path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(files[record.path])
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target, manifest


def load_bundle_directory(bundle_root: Path) -> tuple[SubmissionManifest, dict[str, bytes]]:
    manifest_path = bundle_root / "manifest.json"
    files_dir = bundle_root / "files"
    if not manifest_path.is_file() or not files_dir.is_dir():
        raise FileNotFoundError(f"multi-file bundle is missing manifest.json or files/: {bundle_root}")
    manifest = read_manifest(manifest_path)
    files: dict[str, bytes] = {}
    for record in manifest.files:
        file_path = files_dir / record.path
        if not file_path.is_file():
            raise FileNotFoundError(f"bundle file missing on disk: {file_path}")
        content = file_path.read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha.lower() != record.sha256.lower():
            raise ValueError(f"bundle file hash mismatch for `{record.path}`")
        files[record.path] = content
    actual_bundle_sha = canonical_bundle_sha256(files)
    if manifest.bundle_sha256 and actual_bundle_sha.lower() != manifest.bundle_sha256.lower():
        raise ValueError("manifest bundle_sha256 does not match file contents")
    return manifest, files


def bundle_files_from_archive(data: bytes, *, archive_name: str = "") -> dict[str, bytes]:
    lowered = archive_name.lower()
    if lowered.endswith((".tar.gz", ".tgz")):
        return _files_from_tar_gz(data)
    if lowered.endswith(".zip"):
        return _files_from_zip(data)
    # Sniff by magic when filename is absent.
    if data[:2] == b"PK":
        return _files_from_zip(data)
    return _files_from_tar_gz(data)


def _files_from_tar_gz(data: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if member.issym() or member.islnk():
                raise ValueError(f"bundle archive must not contain symlinks: {member.name}")
            path = normalize_relative_path(member.name)
            files[path] = archive.extractfile(member).read()  # type: ignore[union-attr]
    return normalize_bundle_files(files)


def _files_from_zip(data: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            if info.flag_bits & 0x800:
                name = info.filename
            else:
                name = info.filename
            external = info.external_attr >> 16
            if external & 0o120000 == 0o120000:
                raise ValueError(f"bundle archive must not contain symlinks: {name}")
            path = normalize_relative_path(name)
            files[path] = archive.read(info)
    return normalize_bundle_files(files)


def unified_bundle_diff(*, base_files: dict[str, bytes], submitted_files: dict[str, bytes]) -> str:
    import difflib

    base = normalize_bundle_files(base_files)
    submitted = normalize_bundle_files(submitted_files)
    chunks: list[str] = []
    for path in sorted(set(base) | set(submitted)):
        base_lines = base.get(path, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        submitted_lines = submitted.get(path, b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        if base_lines == submitted_lines:
            continue
        chunks.append(
            "".join(
                difflib.unified_diff(
                    base_lines,
                    submitted_lines,
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        )
    return "".join(chunks)


def bundle_entrypoint_path(bundle_root: Path, manifest: SubmissionManifest | None = None) -> Path:
    manifest = manifest or read_manifest(bundle_root / "manifest.json")
    return bundle_root / "files" / manifest.entrypoint


def is_multi_file_bundle(bundle_root: Path) -> bool:
    return (bundle_root / "manifest.json").is_file() and (bundle_root / "files").is_dir()


def default_ninja_repo_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "ninja"


def total_bundle_bytes(files: dict[str, bytes]) -> int:
    return sum(len(content) for content in files.values())


def enforce_bundle_size_limit(files: dict[str, bytes], *, max_bytes: int = MAX_BUNDLE_BYTES) -> None:
    total = total_bundle_bytes(files)
    if total > max_bytes:
        raise ValueError(f"bundle is {total} bytes; maximum is {max_bytes} bytes")


def collect_harness_py_files(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        raise FileNotFoundError(f"harness directory does not exist: {root}")
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in HARNESS_EXCLUDE_DIR_NAMES or part.startswith(".") for part in relative.parts):
            continue
        files[relative.as_posix()] = path.read_bytes()
    if DEFAULT_ENTRYPOINT not in files:
        raise ValueError(f"harness directory is missing required entrypoint `{DEFAULT_ENTRYPOINT}`")
    return normalize_bundle_files(files)


def collect_harness_from_directory(path: Path) -> dict[str, bytes]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != DEFAULT_ENTRYPOINT:
            raise ValueError(f"single-file harness submissions must be named `{DEFAULT_ENTRYPOINT}`")
        return {DEFAULT_ENTRYPOINT: resolved.read_bytes()}
    return collect_harness_py_files(resolved)


def read_harness_from_git_repo(*, repo: Path, ref: str = "main") -> dict[str, bytes]:
    repo = repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise FileNotFoundError(f"git harness repo does not exist: {repo}")
    remote_ref = f"origin/{ref}"
    fetch_result = subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", "origin", ref],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if fetch_result.returncode != 0:
        detail = (fetch_result.stderr or fetch_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base harness fetch failed for {repo} {ref}: {detail}")
    list_result = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", remote_ref],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if list_result.returncode != 0:
        detail = (list_result.stderr or list_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base harness listing failed for {repo} {ref}: {detail}")
    files: dict[str, bytes] = {}
    for raw_path in list_result.stdout.splitlines():
        path = raw_path.strip()
        if not path or path.endswith("/") or not path.endswith(".py"):
            continue
        if any(part in HARNESS_EXCLUDE_DIR_NAMES or part.startswith(".") for part in PurePosixPath(path).parts):
            continue
        show_result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{remote_ref}:{path}"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if show_result.returncode != 0:
            detail = (show_result.stderr or show_result.stdout or b"").decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"base harness read failed for {remote_ref}:{path}: {detail}")
        files[path] = show_result.stdout
    if DEFAULT_ENTRYPOINT not in files:
        raise RuntimeError(f"base harness at {repo}@{ref} is missing `{DEFAULT_ENTRYPOINT}`")
    return normalize_bundle_files(files)


def entrypoint_sha256(files: dict[str, bytes], *, entrypoint: str = DEFAULT_ENTRYPOINT) -> str:
    normalized = normalize_bundle_files(files)
    if entrypoint not in normalized:
        raise ValueError(f"bundle is missing required entrypoint `{entrypoint}`")
    return hashlib.sha256(normalized[entrypoint]).hexdigest()


def changed_file_statuses(
    *,
    base_files: dict[str, bytes],
    submitted_files: dict[str, bytes],
) -> list[dict[str, str]]:
    base = normalize_bundle_files(base_files)
    submitted = normalize_bundle_files(submitted_files)
    statuses: list[dict[str, str]] = []
    for path in sorted(set(base) | set(submitted)):
        if path not in base:
            statuses.append({"filename": path, "status": "added"})
        elif path not in submitted:
            statuses.append({"filename": path, "status": "removed"})
        elif base[path] != submitted[path]:
            statuses.append({"filename": path, "status": "modified"})
    return statuses
