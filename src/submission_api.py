from __future__ import annotations

import cgi
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from private_submission import (
    AGENT_ENTRYPOINT_FILENAME,
    SubmissionCheck,
    accepted_private_submission_identity,
    agent_bundle_sha256,
    agent_files_violations,
    build_public_submissions_api_payload,
    check_and_record_private_submission_attempt,
    derive_submission_id,
    normalize_agent_files,
    private_submission_check_passed,
    private_submission_registration_check,
    private_submission_signature_payload,
    record_private_submission_acceptance,
    registration_check_is_existing_acceptance,
    run_private_submission_checks,
    write_private_submission_bundle,
)
from r2 import publish_submissions_api_data
from solve_spend import build_solve_spend_payload
from validate import (
    _current_registration_block,
    _open_subtensor,
    _verified_submission_identity_from_config,
    _verify_hotkey_signature,
)

log = logging.getLogger("swe-eval.submission-api")
MAX_REQUEST_BYTES = 5_000_000
MAX_AGENT_BYTES = 5_000_000
MAX_CONCURRENT_SUBMISSIONS = 2
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 6
RATE_LIMIT_MAX_FAILURES = 3
HOTKEY_RATE_LIMIT_WINDOW_SECONDS = 86_400
HOTKEY_RATE_LIMIT_MAX_ATTEMPTS = 4

_submission_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SUBMISSIONS)
_rate_lock = threading.Lock()
_hotkey_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[tuple[float, bool]]] = {}
_public_payload_cache_lock = threading.Lock()
_public_payload_rebuild_lock = threading.Lock()
_public_payload_cache: dict[str, Any] = {"payload": None, "ts": 0.0, "ledger_mtime": None}
PUBLIC_SUBMISSIONS_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class SubmissionApiConfig:
    private_submission_root: Path
    base_agent: Path
    run_config: Any
    judge: Any
    judge_min_score: int
    base_agent_git_repo: Path | None = None
    base_agent_git_ref: str = "main"
    base_agent_git_path: str = "agent.py"
    overwrite: bool = False
    max_request_bytes: int = MAX_REQUEST_BYTES
    max_agent_bytes: int = MAX_AGENT_BYTES
    rate_limit_window_seconds: int = RATE_LIMIT_WINDOW_SECONDS
    rate_limit_max_requests: int = RATE_LIMIT_MAX_REQUESTS
    rate_limit_max_failures: int = RATE_LIMIT_MAX_FAILURES
    hotkey_rate_limit_window_seconds: int = HOTKEY_RATE_LIMIT_WINDOW_SECONDS
    hotkey_rate_limit_max_attempts: int = HOTKEY_RATE_LIMIT_MAX_ATTEMPTS


def _accepted_submissions_ledger_mtime(root: Path) -> float | None:
    path = root / "_accepted_submissions.json"
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def cached_public_submissions_api_payload(*, root: Path) -> dict[str, Any]:
    now = time.monotonic()
    ledger_mtime = _accepted_submissions_ledger_mtime(root)
    with _public_payload_cache_lock:
        cached = _public_payload_cache.get("payload")
        if isinstance(cached, dict) and (now - float(_public_payload_cache["ts"])) < PUBLIC_SUBMISSIONS_CACHE_TTL_SECONDS:
            if _public_payload_cache.get("ledger_mtime") == ledger_mtime:
                return cached
    with _public_payload_rebuild_lock:
        now = time.monotonic()
        ledger_mtime = _accepted_submissions_ledger_mtime(root)
        with _public_payload_cache_lock:
            cached = _public_payload_cache.get("payload")
            if isinstance(cached, dict) and (now - float(_public_payload_cache["ts"])) < PUBLIC_SUBMISSIONS_CACHE_TTL_SECONDS:
                if _public_payload_cache.get("ledger_mtime") == ledger_mtime:
                    return cached
        payload = build_public_submissions_api_payload(root=root)
        with _public_payload_cache_lock:
            _public_payload_cache["payload"] = payload
            _public_payload_cache["ts"] = time.monotonic()
            _public_payload_cache["ledger_mtime"] = ledger_mtime
        return payload


def invalidate_public_submissions_api_cache() -> None:
    with _public_payload_cache_lock:
        _public_payload_cache["payload"] = None
        _public_payload_cache["ts"] = 0.0
        _public_payload_cache["ledger_mtime"] = None


def serve_submissions_api(*, host: str, port: int, config: SubmissionApiConfig) -> None:
    handler = build_handler(config)
    server = ThreadingHTTPServer((host, port), handler)
    log.info("Serving private submissions API on http://%s:%d/api/submissions", host, port)
    server.serve_forever()


def build_handler(config: SubmissionApiConfig):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            normalized_path = parsed.path.rstrip("/")
            if normalized_path == "/api/submissions":
                send_json(self, 200, cached_public_submissions_api_payload(root=config.private_submission_root))
                return
            if normalized_path == "/api/solve-spend":
                send_json(self, 200, solve_spend_payload_for_query(config=config, query=parsed.query))
                return
            send_json(self, 404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/api/submissions":
                send_json(self, 404, {"error": "not_found"})
                return
            client_ip = rate_limit_client_ip(headers=self.headers, client_address=self.client_address)
            if not rate_limit_allowed(client_ip, config=config):
                send_json(self, 429, {"accepted": False, "error": "rate_limited"})
                return
            if request_too_large(self.headers, max_request_bytes=config.max_request_bytes):
                note_rate_result(client_ip, False, config=config)
                send_json(self, 413, {"accepted": False, "error": "request_too_large"})
                return
            if not _submission_slots.acquire(blocking=False):
                send_json(self, 503, {"accepted": False, "error": "submission_api_busy"})
                return
            status, payload = handle_submission_request(headers=self.headers, rfile=self.rfile, config=config)
            try:
                note_rate_result(client_ip, bool(payload.get("accepted")), config=config)
                send_json(self, status, payload)
            finally:
                _submission_slots.release()

        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def solve_spend_payload_for_query(*, config: SubmissionApiConfig, query: str) -> dict[str, Any]:
    values = parse_qs(query, keep_blank_values=False)
    window_seconds = _query_int(values, "window_seconds", default=86_400, minimum=1, maximum=31_536_000)
    return build_solve_spend_payload(tasks_root=config.run_config.tasks_root, window_seconds=window_seconds)


def _query_int(
    values: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(name, [str(default)])[0]
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def handle_submission_request(*, headers: Any, rfile: Any, config: SubmissionApiConfig) -> tuple[int, dict[str, Any]]:
    try:
        form = cgi.FieldStorage(
            fp=rfile,
            headers=headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": headers.get("Content-Type", ""),
                "CONTENT_LENGTH": headers.get("Content-Length", "0"),
            },
        )
        hotkey = form_value(form, "hotkey")
        signature = form_value(form, "signature")
        raw_identity_proof = {
            "username": form_value(form, "agent_username") or form_value(form, "username"),
            "coldkey": form_value(form, "coldkey"),
            "signature": form_value(form, "coldkey_signature") or form_value(form, "coldkeySignature"),
        }
        submitted_id = form_value(form, "submission_id")
        agent_py = form_file_text(form, "agent")
        try:
            extra_files = submitted_extra_files(form)
            agent_files = (
                normalize_agent_files(agent_py=agent_py or None, files=extra_files)
                if agent_py or extra_files
                else {}
            )
        except ValueError as exc:
            return 400, {"accepted": False, "error": f"invalid agent files: {exc}"}
        agent_py = agent_files.get(AGENT_ENTRYPOINT_FILENAME, "")
        if not hotkey or not signature or not agent_py:
            return 400, {"accepted": False, "error": "hotkey, signature, and agent file are required"}
        total_agent_bytes = sum(
            len(path.encode("utf-8")) + len(content.encode("utf-8"))
            for path, content in agent_files.items()
        )
        if total_agent_bytes > config.max_agent_bytes:
            return 413, {"accepted": False, "error": "agent_too_large"}
        agent_sha256 = agent_bundle_sha256(agent_files)
        submission_id = submitted_id or derive_submission_id(
            hotkey=hotkey,
            agent_sha256=agent_sha256,
        )
        signature_payload = private_submission_signature_payload(
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
        )
        signature_valid = _verify_hotkey_signature(hotkey, signature_payload, signature)
        if not signature_valid:
            return 401, precheck_signature_failure_payload(
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                signature_payload=signature_payload,
            )
        registration_block, uid, registration_error = registration_context(
            hotkey=hotkey,
            config=config.run_config,
        )
        registration_check = (
            private_submission_registration_check(
                root=config.private_submission_root,
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                registration_block=registration_block,
            )
            if registration_error is None
            else SubmissionCheck(
                name="Registration Gate",
                status="failed",
                summary="Could not verify the hotkey's current registration.",
                findings=[registration_error],
                metadata={"registration_block": registration_block, "uid": uid},
            )
        )
        if registration_check.status != "passed":
            return 422, precheck_registration_failure_payload(
                signature_valid=signature_valid,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                signature_payload=signature_payload,
                registration_check=registration_check,
                uid=uid,
                registration_block=registration_block,
            )
        if registration_check_is_existing_acceptance(registration_check):
            existing_bundle = config.private_submission_root / submission_id
            existing_identity = accepted_private_submission_identity(
                root=config.private_submission_root,
                submission_id=submission_id,
            )
            publish_submissions_api_data(
                build_public_submissions_api_payload(root=config.private_submission_root)
            )
            return 200, already_accepted_response_payload(
                signature_valid=signature_valid,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                signature_payload=signature_payload,
                bundle_path=existing_bundle,
                uid=uid,
                registration_block=registration_block,
                registration_check=registration_check,
                agent_username=existing_identity["agent_username"] if existing_identity else None,
                coldkey=existing_identity["coldkey"] if existing_identity else None,
            )
        with _hotkey_rate_lock:
            hotkey_rate = check_and_record_private_submission_attempt(
                root=config.private_submission_root,
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                window_seconds=config.hotkey_rate_limit_window_seconds,
                max_attempts=config.hotkey_rate_limit_max_attempts,
            )
        if not bool(hotkey_rate.get("allowed")):
            return 429, hotkey_rate_limit_payload(
                hotkey=hotkey,
                submission_id=submission_id,
                agent_sha256=agent_sha256,
                signature_payload=signature_payload,
                uid=uid,
                registration_block=registration_block,
                hotkey_rate=hotkey_rate,
            )
        base_files = read_base_agent_files(config=config)
        base_agent_py = base_files[AGENT_ENTRYPOINT_FILENAME]
        result = run_private_submission_checks(
            hotkey=hotkey,
            base_agent_py=base_agent_py,
            base_files=base_files,
            openrouter_judge=config.judge,
            min_score=config.judge_min_score,
            submitted_files=agent_files,
        )
        result.checks["registration_gate"] = registration_check
        bundle_path = None
        accepted = bool(result.accepted)
        identity = (
            _verified_submission_identity_from_config(
                config=config.run_config,
                hotkey=hotkey,
                proof=raw_identity_proof,
            )
            if accepted
            else None
        )
        if accepted:
            bundle_path = persist_accepted_submission(
                root=config.private_submission_root,
                submission_id=submission_id,
                hotkey=hotkey,
                agent_files=agent_files,
                result=result,
                signature=signature,
                registration_block=registration_block,
                agent_username=identity["agent_username"] if identity else None,
                coldkey=identity["coldkey"] if identity else None,
                coldkey_signature=identity["coldkey_signature"] if identity else None,
                overwrite=config.overwrite,
            )
            if registration_block is not None:
                record_private_submission_acceptance(
                    root=config.private_submission_root,
                    hotkey=hotkey,
                    submission_id=submission_id,
                    agent_sha256=result.agent_sha256,
                    registration_block=registration_block,
                    agent_username=identity["agent_username"] if identity else None,
                    coldkey=identity["coldkey"] if identity else None,
                    coldkey_signature=identity["coldkey_signature"] if identity else None,
                    uid=uid,
                    validator_state_path=config.run_config.validate_root / "state.json",
                    validate_queue_size=config.run_config.validate_queue_size,
                )
            invalidate_public_submissions_api_cache()
            publish_submissions_api_data(build_public_submissions_api_payload(root=config.private_submission_root))

        payload = response_payload(
            accepted=accepted,
            signature_valid=signature_valid,
            submission_id=submission_id,
            result=result,
            signature_payload=signature_payload,
            bundle_path=bundle_path,
            uid=uid,
            registration_block=registration_block,
            agent_username=identity["agent_username"] if identity else None,
            coldkey=identity["coldkey"] if identity else None,
        )
        return (200 if accepted else 422), payload
    except Exception as exc:
        log.exception("private submission request failed")
        return 500, {"accepted": False, "error": str(exc)}


def read_base_agent_py(*, config: SubmissionApiConfig) -> str:
    return read_base_agent_files(config=config)[AGENT_ENTRYPOINT_FILENAME]


def read_base_agent_files(*, config: SubmissionApiConfig) -> dict[str, str]:
    if config.base_agent_git_repo is None:
        return read_local_base_agent_files(config.base_agent)
    return fetch_git_base_agent_files(
        repo=config.base_agent_git_repo.expanduser(),
        ref=config.base_agent_git_ref,
        path=config.base_agent_git_path,
    )


def read_local_base_agent_files(path: Path) -> dict[str, str]:
    candidate = path.expanduser()
    if candidate.is_file() and candidate.name != AGENT_ENTRYPOINT_FILENAME:
        return {AGENT_ENTRYPOINT_FILENAME: candidate.read_text(encoding="utf-8")}
    root = candidate if candidate.is_dir() else candidate.parent
    if not root.is_dir():
        raise ValueError(f"base agent path does not exist: {candidate}")
    files: dict[str, str] = {}
    for file_path in sorted(root.rglob("*.py")):
        relative = file_path.relative_to(root)
        if any(part == "__pycache__" or part.startswith(".") for part in relative.parts):
            continue
        files[relative.as_posix()] = file_path.read_text(encoding="utf-8")
    if candidate.is_file() and AGENT_ENTRYPOINT_FILENAME not in files:
        files[AGENT_ENTRYPOINT_FILENAME] = candidate.read_text(encoding="utf-8")
    violations = agent_files_violations(files)
    if violations:
        raise ValueError(f"base agent directory is not a valid harness: {violations[0]}")
    return files


def fetch_git_base_agent_files(*, repo: Path, ref: str, path: str) -> dict[str, str]:
    entrypoint = path.strip("/") or AGENT_ENTRYPOINT_FILENAME
    root = str(Path(entrypoint).parent)
    if root == ".":
        root = ""
    repo_path = repo.expanduser().resolve()
    git_base = ["git", "-C", str(repo_path), "-c", f"safe.directory={repo_path}"]
    fetch_result = subprocess.run(
        [*git_base, "fetch", "--quiet", "origin", ref],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if fetch_result.returncode != 0:
        detail = (fetch_result.stderr or fetch_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base agent fetch failed for {repo} {ref}: {detail}")
    treeish = f"origin/{ref}:{root}" if root else f"origin/{ref}"
    ls_result = subprocess.run(
        [*git_base, "ls-tree", "-r", "--name-only", treeish],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if ls_result.returncode != 0:
        detail = (ls_result.stderr or ls_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base agent tree read failed for {treeish}: {detail}")
    files: dict[str, str] = {}
    for rel in sorted(line.strip() for line in ls_result.stdout.splitlines() if line.strip().endswith(".py")):
        if any(part == "__pycache__" or part.startswith(".") for part in Path(rel).parts):
            continue
        full_path = f"{root}/{rel}" if root else rel
        show_result = subprocess.run(
            [*git_base, "show", f"origin/{ref}:{full_path}"],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if show_result.returncode != 0:
            detail = (show_result.stderr or show_result.stdout or "").strip()[-500:]
            raise RuntimeError(f"base agent read failed for origin/{ref}:{full_path}: {detail}")
        files[rel] = show_result.stdout
    violations = agent_files_violations(files)
    if violations:
        raise ValueError(f"base agent tree is not a valid harness: {violations[0]}")
    return files


def fetch_git_base_agent_py(*, repo: Path, ref: str, path: str) -> str:
    repo_path = repo.expanduser().resolve()
    git_base = ["git", "-C", str(repo_path), "-c", f"safe.directory={repo_path}"]
    fetch_result = subprocess.run(
        [*git_base, "fetch", "--quiet", "origin", ref],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if fetch_result.returncode != 0:
        detail = (fetch_result.stderr or fetch_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base agent fetch failed for {repo} {ref}: {detail}")
    show_result = subprocess.run(
        [*git_base, "show", f"origin/{ref}:{path}"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if show_result.returncode != 0:
        detail = (show_result.stderr or show_result.stdout or "").strip()[-500:]
        raise RuntimeError(f"base agent read failed for origin/{ref}:{path}: {detail}")
    return show_result.stdout


def persist_accepted_submission(
    *,
    root: Path,
    submission_id: str,
    hotkey: str,
    result: Any,
    signature: str,
    registration_block: int | None,
    agent_username: str | None,
    coldkey: str | None,
    coldkey_signature: str | None,
    overwrite: bool,
    agent_py: str | None = None,
    agent_files: dict[str, str] | None = None,
) -> Path:
    existing_bundle = root / submission_id
    if (
        existing_bundle.exists()
        and not overwrite
        and private_submission_check_passed(
            root,
            submission_id,
            result.agent_sha256,
            hotkey=hotkey,
            signature_verifier=_verify_hotkey_signature,
        )
    ):
        return existing_bundle
    return write_private_submission_bundle(
        root=root,
        submission_id=submission_id,
        hotkey=hotkey,
        agent_py=agent_py,
        agent_files=agent_files,
        check_result=result,
        signature=signature,
        registration_block=registration_block,
        agent_username=agent_username,
        coldkey=coldkey,
        coldkey_signature=coldkey_signature,
        overwrite=overwrite,
    )


def registration_context(*, hotkey: str, config: Any) -> tuple[int | None, int | None, str | None]:
    try:
        with _open_subtensor(config) as subtensor:
            uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(hotkey, config.validate_netuid)
            if uid is None:
                return None, None, f"Hotkey {hotkey} is not registered on netuid {config.validate_netuid}."
            block = _current_registration_block(
                subtensor=subtensor,
                config=config,
                hotkey=hotkey,
                uid=int(uid),
            )
    except Exception as exc:
        return None, None, f"Registration lookup failed: {exc}"
    if block is None:
        return None, int(uid), f"Could not resolve registration block for hotkey {hotkey}."
    return int(block), int(uid), None


def response_payload(
    *,
    accepted: bool,
    signature_valid: bool,
    submission_id: str,
    result: Any,
    signature_payload: bytes,
    bundle_path: Path | None,
    uid: int | None,
    registration_block: int | None,
    agent_username: str | None = None,
    coldkey: str | None = None,
) -> dict[str, Any]:
    ci_checks = {name: check.to_dict() for name, check in result.checks.items()}
    return {
        "accepted": accepted,
        "signature_valid": signature_valid,
        "submission_id": submission_id,
        "agent_sha256": result.agent_sha256,
        "commitment": f"private-submission:{submission_id}:{result.agent_sha256}",
        "agent_username": agent_username,
        "coldkey": coldkey,
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": str(bundle_path) if bundle_path is not None else None,
        "registration": {"uid": uid, "registration_block": registration_block},
        "ci_checks": ci_checks,
        "llm_judge": ci_checks.get("openrouter_judge"),
        "checks": ci_checks,
    }


def already_accepted_response_payload(
    *,
    signature_valid: bool,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
    bundle_path: Path,
    uid: int | None,
    registration_block: int | None,
    registration_check: SubmissionCheck,
    agent_username: str | None = None,
    coldkey: str | None = None,
) -> dict[str, Any]:
    ci_checks = {"registration_gate": registration_check.to_dict()}
    return {
        "accepted": True,
        "already_accepted": True,
        "message": "This exact private submission was already accepted; no CI or LLM checks were rerun.",
        "signature_valid": signature_valid,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
        "agent_username": agent_username,
        "coldkey": coldkey,
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": str(bundle_path),
        "registration": {"uid": uid, "registration_block": registration_block},
        "ci_checks": ci_checks,
        "llm_judge": None,
        "checks": ci_checks,
    }


def hotkey_rate_limit_payload(
    *,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
    uid: int | None,
    registration_block: int | None,
    hotkey_rate: dict[str, Any],
) -> dict[str, Any]:
    check = SubmissionCheck(
        name="Hotkey Submission Rate Limit",
        status="failed",
        summary=(
            f"Hotkey has reached {hotkey_rate.get('max_attempts')} private submission attempts "
            f"in {hotkey_rate.get('window_seconds')} seconds."
        ),
        findings=["Wait for the 24-hour hotkey submission window to roll forward before retrying."],
        metadata={
            "hotkey": hotkey,
            "attempts": hotkey_rate.get("attempts"),
            "max_attempts": hotkey_rate.get("max_attempts"),
            "window_seconds": hotkey_rate.get("window_seconds"),
            "retry_after_seconds": hotkey_rate.get("retry_after_seconds"),
        },
    )
    ci_checks = {"hotkey_rate_limit": check.to_dict()}
    return {
        "accepted": False,
        "error": "hotkey_rate_limited",
        "signature_valid": True,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": None,
        "registration": {"uid": uid, "registration_block": registration_block},
        "rate_limit": hotkey_rate,
        "ci_checks": ci_checks,
        "llm_judge": None,
        "checks": ci_checks,
    }


def precheck_signature_failure_payload(
    *,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
) -> dict[str, Any]:
    signature_check = SubmissionCheck(
        name="Hotkey Signature",
        status="failed",
        summary="Hotkey signature did not verify for this private submission payload.",
        findings=[
            "Sign the exact signature_payload with the submitting miner hotkey before retrying.",
        ],
    )
    ci_checks = {"hotkey_signature": signature_check.to_dict()}
    return {
        "accepted": False,
        "signature_valid": False,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": None,
        "registration": {"uid": None, "registration_block": None},
        "ci_checks": ci_checks,
        "llm_judge": None,
        "checks": ci_checks,
        "hotkey": hotkey,
    }


def precheck_registration_failure_payload(
    *,
    signature_valid: bool,
    submission_id: str,
    agent_sha256: str,
    signature_payload: bytes,
    registration_check: SubmissionCheck,
    uid: int | None,
    registration_block: int | None,
) -> dict[str, Any]:
    ci_checks = {"registration_gate": registration_check.to_dict()}
    return {
        "accepted": False,
        "signature_valid": signature_valid,
        "submission_id": submission_id,
        "agent_sha256": agent_sha256,
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
        "signature_payload": signature_payload.decode("utf-8"),
        "bundle_path": None,
        "registration": {"uid": uid, "registration_block": registration_block},
        "ci_checks": ci_checks,
        "llm_judge": None,
        "checks": ci_checks,
    }


def form_value(form: cgi.FieldStorage, name: str) -> str:
    value = form.getvalue(name)
    return str(value or "").strip()


def form_file_text(form: cgi.FieldStorage, name: str) -> str:
    item = form[name] if name in form else None
    if item is None or not getattr(item, "file", None):
        return ""
    data = item.file.read()
    if isinstance(data, str):
        return data
    return data.decode("utf-8")


def submitted_extra_files(form: cgi.FieldStorage) -> dict[str, str]:
    """Parse the optional `files` field: a JSON object of path -> file content."""
    raw = form_file_text(form, "files") or form_value(form, "files")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"`files` must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("`files` must be a JSON object mapping relative paths to file contents")
    files: dict[str, str] = {}
    for path, content in payload.items():
        if not isinstance(content, str):
            raise ValueError(f"`files` entry `{path}` must map to a string file content")
        files[str(path)] = content
    return files


def request_too_large(headers: Any, *, max_request_bytes: int) -> bool:
    try:
        content_length = int(headers.get("Content-Length", "0"))
    except (TypeError, ValueError):
        return True
    return content_length <= 0 or content_length > max_request_bytes


def rate_limit_client_ip(*, headers: Any, client_address: Any) -> str:
    peer_ip = socket_peer_ip(client_address)
    if not local_proxy_ip(peer_ip):
        return peer_ip
    return forwarded_client_ip(headers) or peer_ip


def socket_peer_ip(client_address: Any) -> str:
    if not client_address:
        return "unknown"
    try:
        return str(client_address[0])
    except (IndexError, TypeError):
        return "unknown"


def local_proxy_ip(value: str) -> bool:
    try:
        parsed = ip_address(value)
    except ValueError:
        return False
    return parsed.is_loopback


def forwarded_client_ip(headers: Any) -> str | None:
    forwarded_for = header_value(headers, "X-Forwarded-For")
    if forwarded_for:
        first_forwarded = forwarded_for.split(",", 1)[0].strip()
        if valid_ip(first_forwarded):
            return first_forwarded

    real_ip = header_value(headers, "X-Real-IP")
    if real_ip and valid_ip(real_ip.strip()):
        return real_ip.strip()

    return None


def header_value(headers: Any, name: str) -> str:
    try:
        return str(headers.get(name, "") or "")
    except AttributeError:
        return ""


def valid_ip(value: str) -> bool:
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def rate_limit_allowed(client_ip: str, *, config: SubmissionApiConfig) -> bool:
    now = time.monotonic()
    with _rate_lock:
        bucket = recent_rate_events(_rate_buckets.get(client_ip, []), now, config=config)
        request_count = len(bucket)
        failure_count = sum(1 for _timestamp, accepted in bucket if not accepted)
        _rate_buckets[client_ip] = bucket
    return request_count < config.rate_limit_max_requests and failure_count < config.rate_limit_max_failures


def note_rate_result(client_ip: str, accepted: bool, *, config: SubmissionApiConfig) -> None:
    now = time.monotonic()
    with _rate_lock:
        bucket = recent_rate_events(_rate_buckets.get(client_ip, []), now, config=config)
        bucket.append((now, accepted))
        _rate_buckets[client_ip] = bucket


def recent_rate_events(
    events: list[tuple[float, bool]],
    now: float,
    *,
    config: SubmissionApiConfig,
) -> list[tuple[float, bool]]:
    return recent_rate_events_for_window(events, now, config.rate_limit_window_seconds)


def recent_rate_events_for_window(
    events: list[tuple[float, bool]],
    now: float,
    window_seconds: int,
) -> list[tuple[float, bool]]:
    cutoff = now - window_seconds
    return [(timestamp, accepted) for timestamp, accepted in events if timestamp >= cutoff]


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
