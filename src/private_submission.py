from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import logging
import py_compile
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_OPENROUTER_MIN_SCORE = 65
PRIVATE_SUBMISSION_ACCEPTANCE_LEDGER = "_accepted_submissions.json"
PRIVATE_SUBMISSION_QUEUE_WAKEUP = "_queue_wakeup"
PRIVATE_SUBMISSION_ATTEMPT_LEDGER = "_submission_attempts.json"
MINER_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,64}$")
AGENT_ENTRYPOINT_FILENAME = "agent.py"
MAX_AGENT_FILES = 32
MAX_AGENT_FILE_PATH_SEGMENTS = 8
_AGENT_FILE_DIR_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}$")
log = logging.getLogger("swe-eval.private-submission")
_AGENT_FILE_BASENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]{0,63}\.py$")
REQUIRED_SOLVE_ARGS = ("repo_path", "issue", "model", "api_base", "api_key")
ALLOWED_ENV_NAMES = {
    "AGENT_MAX_STEPS",
    "AGENT_COMMAND_TIMEOUT",
    "AGENT_MODEL",
    "NINJA_MODEL",
    "AGENT_API_BASE",
    "NINJA_INFERENCE_BASE_URL",
    "OPENAI_BASE_URL",
    "AGENT_API_KEY",
    "NINJA_INFERENCE_API_KEY",
    "OPENAI_API_KEY",
    "AGENT_MAX_TOKENS",
    "AGENT_MAX_OBSERVATION_CHARS",
    "AGENT_MAX_TOTAL_LOG_CHARS",
    # Validator-provided per-round budget, exported into the solver container
    # by docker_solver so agents can pace themselves against the real timeout.
    "TAU_AGENT_TIMEOUT_SECONDS",
}
FORBIDDEN_SAMPLING_NAMES = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_a",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
}
FORBIDDEN_SUBSTRINGS = (
    "openrouter_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "groq_api_key",
    "together_api_key",
    "fireworks_api_key",
    "mistral_api_key",
    "deepinfra_api_key",
    "github_token",
    "api.openai.com",
    "openrouter.ai",
    "anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "api.together.xyz",
    "api.fireworks.ai",
    "api.mistral.ai",
    "api.deepseek.com",
    "deepinfra.com",
    "cohere.ai",
    "/proc/self/environ",
    "/proc/environ",
    ".ssh",
    "id_rsa",
    ".netrc",
    "wallet",
)
PROTECTED_EDIT_MARKERS = (
    "def solve(",
    "repo_path: str,",
    "issue: str,",
    "model: Optional[str] = None,",
    "api_base: Optional[str] = None,",
    "api_key: Optional[str] = None,",
    "def _resolve_inference_config(",
    "DEFAULT_MODEL =",
    "DEFAULT_API_BASE =",
    "DEFAULT_API_KEY =",
    "DEFAULT_TEMPERATURE =",
)
PROTECTED_HUNK_SYMBOLS = ("_resolve_inference_config",)
KNOWN_BASELINE_PYFLAKES_SUBSTRINGS = (
    "local variable '_wall_start' is assigned to but never used",
)


@dataclass(slots=True)
class SubmissionCheck:
    name: str
    status: str
    summary: str
    findings: list[str] = field(default_factory=list)
    score: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PrivateSubmissionCheckResult:
    accepted: bool
    agent_sha256: str
    checks: dict[str, SubmissionCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "agent_sha256": self.agent_sha256,
            "checks": {key: value.to_dict() for key, value in self.checks.items()},
        }


JudgeFn = Callable[[dict[str, Any]], dict[str, Any]]
SignatureVerifier = Callable[[str, bytes, str], bool]


def normalize_agent_files(
    *,
    agent_py: str | None = None,
    files: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge a legacy single-file submission with optional extra module files."""
    merged: dict[str, str] = {}
    for raw_path, raw_content in (files or {}).items():
        merged[str(raw_path).strip()] = str(raw_content)
    if agent_py is not None:
        existing = merged.get(AGENT_ENTRYPOINT_FILENAME)
        if existing is not None and existing != agent_py:
            raise ValueError(
                f"`{AGENT_ENTRYPOINT_FILENAME}` was provided twice with different contents"
            )
        merged[AGENT_ENTRYPOINT_FILENAME] = agent_py
    if not merged:
        raise ValueError("a submission requires at least one agent file")
    return merged


def agent_files_violations(files: dict[str, str]) -> list[str]:
    violations: list[str] = []
    if AGENT_ENTRYPOINT_FILENAME not in files:
        violations.append(
            f"submission must include `{AGENT_ENTRYPOINT_FILENAME}` as the agent entrypoint."
        )
    if len(files) > MAX_AGENT_FILES:
        violations.append(
            f"submission has {len(files)} files; the maximum is {MAX_AGENT_FILES}."
        )
    for path in sorted(files):
        violations.extend(agent_file_path_violations(path))
    return _dedupe(violations)


def agent_file_path_violations(path: str) -> list[str]:
    if not path or path != path.strip():
        return [f"agent file path `{path}` must not be empty or padded with whitespace."]
    if path.startswith("/") or "\\" in path:
        return [f"agent file path `{path}` must be a relative POSIX path."]
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        return [f"agent file path `{path}` must not contain empty, `.` or `..` segments."]
    if len(segments) > MAX_AGENT_FILE_PATH_SEGMENTS:
        return [f"agent file path `{path}` is nested deeper than {MAX_AGENT_FILE_PATH_SEGMENTS} segments."]
    violations: list[str] = []
    for segment in segments[:-1]:
        if not _AGENT_FILE_DIR_SEGMENT_RE.fullmatch(segment):
            violations.append(f"agent file path `{path}` has an invalid directory segment `{segment}`.")
    if not _AGENT_FILE_BASENAME_RE.fullmatch(segments[-1]):
        violations.append(f"agent file `{path}` must be a Python module named like `module.py`.")
    return violations


def agent_bundle_sha256(files: dict[str, str]) -> str:
    """Deterministic submission hash.

    Single-file submissions keep the historical sha256-of-agent.py value so
    existing commitments, signatures, and ledgers stay valid.
    """
    if set(files) == {AGENT_ENTRYPOINT_FILENAME}:
        return hashlib.sha256(files[AGENT_ENTRYPOINT_FILENAME].encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for path in sorted(files):
        content_sha = hashlib.sha256(files[path].encode("utf-8")).hexdigest()
        digest.update(f"{path}\0{content_sha}\n".encode())
    return digest.hexdigest()


def agent_files_manifest(files: dict[str, str]) -> dict[str, str]:
    return {
        path: hashlib.sha256(files[path].encode("utf-8")).hexdigest()
        for path in sorted(files)
    }


def _agent_local_module_roots(files: dict[str, str]) -> frozenset[str]:
    roots: set[str] = set()
    for path in files:
        root = path.split("/", 1)[0]
        if root.endswith(".py"):
            root = root[: -len(".py")]
        if root:
            roots.add(root)
    return frozenset(roots)


def run_private_submission_checks(
    *,
    hotkey: str,
    submitted_agent_py: str | None = None,
    base_agent_py: str,
    base_files: dict[str, str] | None = None,
    openrouter_judge: JudgeFn | None = None,
    min_score: int = DEFAULT_OPENROUTER_MIN_SCORE,
    submitted_files: dict[str, str] | None = None,
) -> PrivateSubmissionCheckResult:
    files = normalize_agent_files(agent_py=submitted_agent_py, files=submitted_files)
    agent_sha = agent_bundle_sha256(files)
    normalized_base_files = normalize_base_agent_files(
        base_agent_py=base_agent_py,
        files=base_files,
    )
    patch = _files_patch(base_files=normalized_base_files, submitted_files=files)
    smoke = run_agent_smoke_checks(files=files)
    scope_guard = run_scope_guard(hotkey=hotkey, files=files, patch=patch)
    checks = {"agent_smoke": smoke, "scope_guard": scope_guard}
    if smoke.status == "passed" and scope_guard.status == "passed":
        checks["openrouter_judge"] = run_openrouter_judge_gate(
            hotkey=hotkey,
            base_agent_py=base_agent_py,
            base_files=normalized_base_files,
            submitted_agent_py=files.get(AGENT_ENTRYPOINT_FILENAME, ""),
            patch=patch,
            judge=openrouter_judge,
            min_score=min_score,
            submitted_files=files,
        )
    else:
        failed = "agent smoke" if smoke.status != "passed" else "scope guard"
        checks["openrouter_judge"] = SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="skipped",
            summary=f"Skipped because {failed} failed.",
        )
    accepted = all(check.status == "passed" for check in checks.values())
    return PrivateSubmissionCheckResult(accepted=accepted, agent_sha256=agent_sha, checks=checks)


def normalize_base_agent_files(
    *,
    base_agent_py: str,
    files: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the public base harness files used for submission diffs.

    Legacy callers pass only `base_agent_py`; repo-aware callers can provide the
    whole public harness file set so omissions/deletions are visible to the LLM
    gate instead of silently comparing only the entrypoint.
    """
    merged = normalize_agent_files(agent_py=base_agent_py, files=files)
    return dict(sorted(merged.items()))


def run_agent_smoke_checks(
    *,
    agent_py: str | None = None,
    files: dict[str, str] | None = None,
) -> SubmissionCheck:
    files = normalize_agent_files(agent_py=agent_py, files=files)
    findings: list[str] = []
    for path in sorted(files):
        findings.extend(_agent_file_smoke_findings(path=path, source=files[path]))
    findings = _dedupe(findings)
    if findings:
        return SubmissionCheck(
            name="Agent Smoke",
            status="failed",
            summary="submitted agent files failed local smoke checks.",
            findings=findings,
        )
    return SubmissionCheck(
        name="Agent Smoke",
        status="passed",
        summary="submitted agent files compile and have no new pyflakes findings.",
    )


def _agent_file_smoke_findings(*, path: str, source: str) -> list[str]:
    findings: list[str] = []
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tmp:
            tmp.write(source)
            tmp_path = Path(tmp.name)
        try:
            py_compile.compile(str(tmp_path), doraise=True)
        finally:
            tmp_path.unlink(missing_ok=True)
    except py_compile.PyCompileError as exc:
        findings.append(f"{path} failed py_compile: {exc.msg}")
    except OSError as exc:
        findings.append(f"{path} smoke check could not create temp file: {exc}")
    findings.extend(_pyflakes_findings(source, filename=path))
    return findings


def run_scope_guard(
    *,
    hotkey: str,
    agent_py: str | None = None,
    patch: str | None = None,
    files: dict[str, str] | None = None,
) -> SubmissionCheck:
    files = normalize_agent_files(agent_py=agent_py, files=files)
    local_modules = _agent_local_module_roots(files)
    findings: list[str] = []
    findings.extend(agent_files_violations(files))
    if patch is not None:
        findings.extend(_agent_patch_violations(patch))
    for path in sorted(files):
        findings.extend(
            _agent_source_violations(
                files[path],
                filename=path,
                require_solve=path == AGENT_ENTRYPOINT_FILENAME,
                local_modules=local_modules,
            )
        )
    findings = _dedupe(findings)
    if findings:
        return SubmissionCheck(
            name="Submission Scope Guard",
            status="failed",
            summary="submitted agent files failed the local submission contract checks.",
            findings=findings,
        )
    return SubmissionCheck(
        name="Submission Scope Guard",
        status="passed",
        summary="submitted agent files satisfy the local submission contract checks.",
    )


def run_openrouter_judge_gate(
    *,
    hotkey: str,
    base_agent_py: str,
    base_files: dict[str, str],
    submitted_agent_py: str,
    patch: str,
    judge: JudgeFn | None,
    min_score: int,
    submitted_files: dict[str, str] | None = None,
) -> SubmissionCheck:
    if judge is None:
        return SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="skipped",
            summary="No local OpenRouter judge function was configured.",
        )
    files = normalize_agent_files(agent_py=submitted_agent_py, files=submitted_files)
    changed_files = [
        {
            "filename": path,
            "status": (
                "deleted"
                if path not in files
                else "added"
                if path not in base_files
                else "modified"
            ),
        }
        for path in sorted(set(base_files) | set(files))
        if base_files.get(path) != files.get(path)
    ]
    payload = {
        "hotkey": hotkey,
        "title": f"{hotkey} private submission",
        "changed_files": changed_files,
        "static_findings": {
            "fail_reasons": [],
            "warnings": [],
            "findings": [],
            "changed_files": sorted(files),
        },
        "patch": patch,
        "base_agent_py": base_agent_py,
        "base_files": base_files,
        "submitted_agent_py": submitted_agent_py,
        "submitted_files": files,
    }
    try:
        judgment = judge(payload)
    except Exception as exc:
        return SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="failed",
            summary=f"OpenRouter judge failed: {exc}",
        )
    verdict = str(judgment.get("verdict", "fail")).lower()
    score = _coerce_score(judgment.get("overall_score"))
    findings = [str(item) for item in judgment.get("reasons") or []]
    score_failures = judge_score_failures(judgment, min_score=min_score)
    if verdict == "fail" or score_failures:
        findings.extend(score_failures)
        findings.append(f"Judge verdict={verdict}, score={score}, threshold={min_score}.")
        return SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="failed",
            summary=str(judgment.get("summary") or "OpenRouter judge rejected the submission."),
            findings=_dedupe(findings),
            score=score,
            metadata={"judgment": judgment},
        )
    status = "warn" if verdict == "warn" else "passed"
    return SubmissionCheck(
        name="OpenRouter Submission Judge",
        status=status,
        summary=str(judgment.get("summary") or "OpenRouter judge accepted the submission."),
        findings=findings,
        score=score,
        metadata={"judgment": judgment},
    )


def judge_score_failures(judgment: dict[str, Any], *, min_score: int) -> list[str]:
    failures: list[str] = []
    overall = _coerce_score(judgment.get("overall_score"))
    if overall < min_score:
        failures.append(f"overall_score={overall} is below required minimum {min_score}.")

    failures.extend(_judge_risk_failures(judgment, overall=overall, min_score=min_score))

    component_floor = max(0, min(100, int(min_score) - 5))
    real_edit_floor = max(0, min(100, int(min_score)))
    for name in ("real_edit_score", "safety_score", "scope_score", "contract_score"):
        if name not in judgment:
            continue
        score = _coerce_score(judgment.get(name))
        floor = real_edit_floor if name == "real_edit_score" else component_floor
        if score < floor:
            failures.append(f"{name}={score} is below component minimum {floor}.")

    if "real_edit_score" in judgment:
        real_edit = _coerce_score(judgment.get("real_edit_score"))
        if overall > real_edit + 10:
            failures.append(
                f"overall_score={overall} exceeds real_edit_score={real_edit} by more than 10 points."
            )
    return failures


def _judge_risk_failures(judgment: dict[str, Any], *, overall: int, min_score: int) -> list[str]:
    risk_categories = _judge_risk_categories(judgment.get("risks") or [])
    named_cosmetic_or_goodhart = bool(risk_categories & _BORDERLINE_NON_CONTRIBUTION_RISK_CATEGORIES)
    borderline = overall <= min(100, int(min_score) + 5)
    if named_cosmetic_or_goodhart and borderline:
        return [
            "judge reported cosmetic-copy/goodhart/non-contribution risk on a borderline score; requires a clearly functional contribution."
        ]
    return []


def _judge_risk_categories(risks: Any) -> frozenset[str]:
    if not isinstance(risks, list):
        return frozenset()
    return frozenset(
        category
        for item in risks
        for category in [_judge_risk_category(item)]
        if category is not None
    )


def _judge_risk_category(item: Any) -> str | None:
    if isinstance(item, dict):
        return _normalize_judge_risk_category(
            item.get("category") or item.get("type") or item.get("name") or item.get("risk")
        )
    if isinstance(item, str):
        return _normalize_judge_risk_category(_risk_label_from_text(item))
    return None


def _risk_label_from_text(text: str) -> str:
    return re.split(r":|\s+(?:-|--|—)\s+", text.strip(), maxsplit=1)[0]


def _normalize_judge_risk_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return _JUDGE_RISK_CATEGORY_ALIASES.get(normalized, normalized or None)


_JUDGE_RISK_CATEGORY_ALIASES = {
    "cosmetic-copy": "cosmetic-copy",
    "cosmetic": "cosmetic-copy",
    "goodhart": "goodhart",
    "goodharting": "goodhart",
    "comment-churn": "comment-churn",
    "newline-normalization": "newline-normalization",
    "normalizes-newlines": "newline-normalization",
    "parameter-only": "parameter-only",
}
_BORDERLINE_NON_CONTRIBUTION_RISK_CATEGORIES = frozenset(
    {"cosmetic-copy", "goodhart", "comment-churn", "newline-normalization", "parameter-only"}
)


def write_private_submission_bundle(
    *,
    root: Path,
    submission_id: str,
    hotkey: str,
    agent_py: str | None = None,
    check_result: PrivateSubmissionCheckResult,
    signature: str,
    registration_block: int | None = None,
    agent_username: str | None = None,
    coldkey: str | None = None,
    coldkey_signature: str | None = None,
    overwrite: bool = False,
    agent_files: dict[str, str] | None = None,
) -> Path:
    if not valid_submission_id(submission_id):
        raise ValueError("submission_id must contain only letters, numbers, '.', '_' or '-'")
    files = normalize_agent_files(agent_py=agent_py, files=agent_files)
    path_violations = agent_files_violations(files)
    if path_violations:
        raise ValueError(f"private submission has invalid agent files: {path_violations[0]}")
    target = root / submission_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"private submission already exists: {submission_id}")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(files):
        file_path = target / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(files[path], encoding="utf-8")
    check_payload = check_result.to_dict()
    ci_checks = check_payload["checks"]
    (target / "check_result.json").write_text(
        json.dumps(
            _compact_none(
                {
                    "submission_id": submission_id,
                    "hotkey": hotkey,
                    "registration_block": registration_block,
                    "signature": signature,
                    "agent_files": agent_files_manifest(files),
                    **_verified_identity_metadata(
                        agent_username=agent_username,
                        coldkey=coldkey,
                        coldkey_signature=coldkey_signature,
                    ),
                    "signature_payload": private_submission_signature_payload(
                        hotkey=hotkey,
                        submission_id=submission_id,
                        agent_sha256=check_result.agent_sha256,
                    ).decode("utf-8"),
                    **check_payload,
                    "ci_checks": ci_checks,
                    "llm_judge": ci_checks.get("openrouter_judge"),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def private_submission_registration_check(
    *,
    root: Path,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    registration_block: int | None,
) -> SubmissionCheck:
    if registration_block is None:
        return SubmissionCheck(
            name="Registration Gate",
            status="failed",
            summary="Could not resolve the hotkey's current registration block.",
            findings=["Registration block is required to enforce one accepted private submission per registration."],
        )

    current_registration = int(registration_block)
    ledger = _read_acceptance_ledger(root)
    existing = ledger.get("hotkeys", {}).get(hotkey)
    if isinstance(existing, dict):
        existing_registration = _optional_int(existing.get("registration_block"))
        same_submission = (
            str(existing.get("submission_id") or "") == submission_id
            and str(existing.get("agent_sha256") or "").lower() == agent_sha256.lower()
        )
        if existing_registration is None or existing_registration >= current_registration:
            if same_submission:
                return SubmissionCheck(
                    name="Registration Gate",
                    status="passed",
                    summary="This exact private submission is already accepted for the current registration.",
                    metadata={
                        "registration_block": current_registration,
                        "prior_submission_id": submission_id,
                        "prior_agent_sha256": agent_sha256.lower(),
                    },
                )
            prior_id = str(existing.get("submission_id") or "unknown")
            prior_sha = str(existing.get("agent_sha256") or "unknown")
            return SubmissionCheck(
                name="Registration Gate",
                status="failed",
                summary="Hotkey already has one accepted private submission for this registration.",
                findings=[
                    f"Prior accepted submission `{prior_id}` at registration block {existing_registration}; "
                    "the hotkey must re-register before another private submission can be accepted."
                ],
                metadata={
                    "registration_block": current_registration,
                    "prior_registration_block": existing_registration,
                    "prior_submission_id": prior_id,
                    "prior_agent_sha256": prior_sha,
                },
            )

    return SubmissionCheck(
        name="Registration Gate",
        status="passed",
        summary="Hotkey has no accepted private submission for the current registration.",
        metadata={"registration_block": current_registration},
    )


def registration_check_is_existing_acceptance(check: SubmissionCheck) -> bool:
    return (
        check.status == "passed"
        and str(check.metadata.get("prior_submission_id") or "") != ""
        and str(check.metadata.get("prior_agent_sha256") or "") != ""
    )


def accepted_private_submission_identity(*, root: Path, submission_id: str) -> dict[str, str] | None:
    payload = _read_bundle_check_result(root=root, submission_id=submission_id)
    if payload.get("agent_identity_verified") is not True:
        return None
    agent_username = str(payload.get("agent_username") or "").strip()
    coldkey = str(payload.get("coldkey") or "").strip()
    if not agent_username or not coldkey:
        return None
    return {"agent_username": agent_username, "coldkey": coldkey}


def record_private_submission_acceptance(
    *,
    root: Path,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    registration_block: int,
    agent_username: str | None = None,
    coldkey: str | None = None,
    coldkey_signature: str | None = None,
    uid: int | None = None,
    validator_state_path: Path | None = None,
    validate_queue_size: int | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    accepted_at = datetime.now(UTC).isoformat()
    ledger = _read_acceptance_ledger(root)
    ledger.setdefault("version", 1)
    hotkeys = ledger.setdefault("hotkeys", {})
    if not isinstance(hotkeys, dict):
        raise ValueError("private submission acceptance ledger has invalid `hotkeys`")
    entry = {
        "registration_block": int(registration_block),
        "submission_id": submission_id,
        "agent_sha256": agent_sha256.lower(),
        "accepted_at": accepted_at,
        **_verified_identity_metadata(
            agent_username=agent_username,
            coldkey=coldkey,
            coldkey_signature=coldkey_signature,
        ),
    }
    if uid is not None:
        entry["uid"] = int(uid)
    hotkeys[hotkey] = entry
    _write_acceptance_ledger(root, ledger)
    if validator_state_path is not None and uid is not None:
        from validator_state_io import (
            enqueue_private_submission_in_state,
            private_submission_validator_queue_entry,
        )

        queue_entry = private_submission_validator_queue_entry(
            hotkey=hotkey,
            submission_id=submission_id,
            agent_sha256=agent_sha256,
            registration_block=registration_block,
            uid=int(uid),
            accepted_at=accepted_at,
            agent_username=agent_username,
            coldkey=coldkey,
            coldkey_signature=coldkey_signature,
        )
        try:
            enqueue_private_submission_in_state(
                state_path=validator_state_path,
                submission=queue_entry,
                queue_size_limit=validate_queue_size,
            )
        except Exception:
            log.exception(
                "Atomic validator-state enqueue failed for submission %s (ledger recorded)",
                submission_id,
            )
    touch_private_submission_queue_wakeup(root=root)


def backfill_acceptance_ledger_uids(*, root: Path, netuid: int = 66) -> int:
    """Write missing uids into the acceptance ledger from chain state."""
    from hotkey_uid_cache import clear_hotkey_uid_cache, hotkey_uid_map

    ledger = _read_acceptance_ledger(root)
    hotkeys = ledger.get("hotkeys", {})
    if not isinstance(hotkeys, dict):
        return 0

    clear_hotkey_uid_cache()
    uid_map = hotkey_uid_map(netuid=netuid, ttl_seconds=0)
    updated = 0
    for hotkey, entry in hotkeys.items():
        if not isinstance(entry, dict) or entry.get("uid") is not None:
            continue
        uid = uid_map.get(str(hotkey))
        if uid is None:
            continue
        entry["uid"] = int(uid)
        updated += 1

    if updated:
        _write_acceptance_ledger(root, ledger)
        touch_private_submission_queue_wakeup(root=root)
    return updated


def touch_private_submission_queue_wakeup(*, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / PRIVATE_SUBMISSION_QUEUE_WAKEUP
    path.write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")


def check_and_record_private_submission_attempt(
    *,
    root: Path,
    hotkey: str,
    submission_id: str,
    agent_sha256: str,
    window_seconds: int = 86_400,
    max_attempts: int = 4,
    now: datetime | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(UTC)
    ledger = _read_attempt_ledger(root)
    ledger.setdefault("version", 1)
    hotkeys = ledger.setdefault("hotkeys", {})
    if not isinstance(hotkeys, dict):
        raise ValueError("private submission attempt ledger has invalid `hotkeys`")

    recent_attempts = recent_private_submission_attempts(
        hotkeys.get(hotkey, []),
        now=current_time,
        window_seconds=window_seconds,
    )
    if len(recent_attempts) >= max_attempts:
        hotkeys[hotkey] = recent_attempts
        _write_attempt_ledger(root, ledger)
        oldest_attempted_at = _parse_attempted_at(recent_attempts[0])
        retry_after_seconds = (
            max(0, int(window_seconds - (current_time - oldest_attempted_at).total_seconds()))
            if oldest_attempted_at is not None
            else int(window_seconds)
        )
        return {
            "allowed": False,
            "attempts": len(recent_attempts),
            "max_attempts": int(max_attempts),
            "window_seconds": int(window_seconds),
            "retry_after_seconds": retry_after_seconds,
        }

    hotkeys[hotkey] = [
        *recent_attempts,
        {
            "submission_id": submission_id,
            "agent_sha256": agent_sha256.lower(),
            "attempted_at": current_time.isoformat(),
        },
    ]
    _write_attempt_ledger(root, ledger)
    return {
        "allowed": True,
        "attempts": len(hotkeys[hotkey]),
        "max_attempts": int(max_attempts),
        "window_seconds": int(window_seconds),
        "retry_after_seconds": 0,
    }


def recent_private_submission_attempts(
    attempts: Any,
    *,
    now: datetime,
    window_seconds: int,
) -> list[dict[str, Any]]:
    if not isinstance(attempts, list):
        return []
    cutoff = now.timestamp() - int(window_seconds)
    return [
        attempt
        for attempt in attempts
        if isinstance(attempt, dict)
        for attempted_at in [_parse_attempted_at(attempt)]
        if attempted_at is not None and attempted_at.timestamp() >= cutoff
    ]


def build_public_submissions_api_payload(*, root: Path) -> dict[str, Any]:
    ledger = _read_acceptance_ledger(root)
    hotkeys = ledger.get("hotkeys", {})
    submissions = [
        public_submission
        for hotkey, entry in sorted(hotkeys.items(), key=lambda item: str(item[0]))
        if isinstance(entry, dict)
        for public_submission in [_public_submission_from_ledger_entry(root=root, hotkey=str(hotkey), entry=entry)]
        if public_submission is not None
    ]
    return {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "submissions": submissions,
    }


def accepted_private_submission_entries(*, root: Path) -> list[dict[str, Any]]:
    ledger = _read_acceptance_ledger(root)
    hotkeys = ledger.get("hotkeys", {})
    if not isinstance(hotkeys, dict):
        return []
    return [
        _compact_none(
            {
                "hotkey": str(hotkey),
                "submission_id": str(entry.get("submission_id") or ""),
                "agent_sha256": str(entry.get("agent_sha256") or "").lower(),
                "registration_block": _optional_int(entry.get("registration_block")),
                "accepted_at": entry.get("accepted_at"),
                "uid": _optional_int(entry.get("uid")),
                "agent_username": entry.get("agent_username") or entry.get("username"),
                "coldkey": entry.get("coldkey"),
                "coldkey_signature": entry.get("coldkey_signature") or entry.get("signature"),
            }
        )
        for hotkey, entry in sorted(hotkeys.items(), key=lambda item: str(item[0]))
        if isinstance(entry, dict)
    ]


def _public_submission_from_ledger_entry(
    *,
    root: Path,
    hotkey: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    submission_id = str(entry.get("submission_id") or "")
    agent_sha256 = str(entry.get("agent_sha256") or "").lower()
    check_result = _read_bundle_check_result(root=root, submission_id=submission_id)
    if not submission_id or not agent_sha256:
        return None
    return _compact_none(
        {
            "submission_id": submission_id,
            "hotkey": hotkey,
            "agent_sha256": agent_sha256,
            "commitment": f"private-submission:{submission_id}:{agent_sha256}",
            "registration_block": _optional_int(entry.get("registration_block")),
            "uid": _optional_int(entry.get("uid")),
            "accepted_at": entry.get("accepted_at"),
            **_public_identity_metadata(entry),
            "accepted": bool(check_result.get("accepted", True)),
            "ci_checks": _public_ci_checks(check_result.get("ci_checks") or check_result.get("checks")),
            "llm_judge": _public_check(check_result.get("llm_judge")),
        }
    )


def _verified_identity_metadata(
    *,
    agent_username: str | None,
    coldkey: str | None,
    coldkey_signature: str | None,
) -> dict[str, Any]:
    if not agent_username or not coldkey or not coldkey_signature:
        return {}
    return {
        "agent_identity_verified": True,
        "agent_username": agent_username,
        "coldkey": coldkey,
        "coldkey_signature": coldkey_signature,
    }


def _public_identity_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("agent_identity_verified") is not True:
        return {}
    return _compact_none(
        {
            "agent_username": entry.get("agent_username") or entry.get("username"),
            "coldkey": entry.get("coldkey"),
        }
    )


def _read_bundle_check_result(*, root: Path, submission_id: str) -> dict[str, Any]:
    if not valid_submission_id(submission_id):
        return {}
    path = root / submission_id / "check_result.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _public_ci_checks(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): public_check
        for name, check in sorted(value.items(), key=lambda item: str(item[0]))
        for public_check in [_public_check(check)]
        if public_check is not None
    }


def _public_check(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    metadata = value.get("metadata")
    judgment = metadata.get("judgment") if isinstance(metadata, dict) else None
    return _compact_none(
        {
            "name": value.get("name"),
            "status": value.get("status"),
            "summary": value.get("summary"),
            "findings": value.get("findings"),
            "score": value.get("score"),
            "judgment": _public_judgment(judgment),
        }
    )


def _public_judgment(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return _compact_none(
        {
            "verdict": value.get("verdict"),
            "overall_score": value.get("overall_score"),
            "summary": value.get("summary"),
            "reasons": value.get("reasons"),
        }
    )


def derive_submission_id(*, hotkey: str, agent_sha256: str) -> str:
    safe_hotkey = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe_hotkey}-{agent_sha256.lower()[:16]}"


def private_submission_bundle_files(*, root: Path, submission_id: str) -> dict[str, str] | None:
    """Return the verified agent files recorded in a bundle's manifest.

    Returns None when the bundle, its manifest, or any per-file hash does not
    line up with what is on disk. Legacy bundles without a manifest report
    just their agent.py.
    """
    if not valid_submission_id(submission_id):
        return None
    bundle = root / submission_id
    result = _read_bundle_check_result(root=root, submission_id=submission_id)
    manifest = result.get("agent_files")
    if not isinstance(manifest, dict) or not manifest:
        agent_path = bundle / AGENT_ENTRYPOINT_FILENAME
        if not agent_path.is_file():
            return None
        try:
            return {AGENT_ENTRYPOINT_FILENAME: agent_path.read_bytes().decode("utf-8")}
        except (OSError, UnicodeDecodeError):
            return None
    files: dict[str, str] = {}
    for raw_path, expected_file_sha in manifest.items():
        path = str(raw_path)
        if agent_file_path_violations(path):
            return None
        file_path = bundle / path
        if not file_path.is_file():
            return None
        try:
            content = file_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != str(expected_file_sha).lower():
            return None
        files[path] = content
    if AGENT_ENTRYPOINT_FILENAME not in files:
        return None
    for stray in bundle.rglob("*.py"):
        if stray.relative_to(bundle).as_posix() not in files:
            return None
    return files


def private_submission_check_passed(
    root: Path,
    submission_id: str,
    expected_sha256: str,
    *,
    hotkey: str,
    signature_verifier: SignatureVerifier,
) -> bool:
    bundle = root / submission_id
    agent_path = bundle / "agent.py"
    result_path = bundle / "check_result.json"
    if not agent_path.is_file() or not result_path.is_file():
        return False
    files = private_submission_bundle_files(root=root, submission_id=submission_id)
    if files is None:
        return False
    actual_sha = agent_bundle_sha256(files)
    if actual_sha.lower() != expected_sha256.lower():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not bool(result.get("accepted")) or str(result.get("agent_sha256", "")).lower() != actual_sha:
        return False
    if str(result.get("hotkey") or "") != str(hotkey):
        return False
    signature = str(result.get("signature") or "").strip()
    if not signature:
        return False
    payload = private_submission_signature_payload(
        hotkey=hotkey,
        submission_id=submission_id,
        agent_sha256=actual_sha,
    )
    return bool(signature_verifier(hotkey, payload, signature))


def private_submission_signature_payload(*, hotkey: str, submission_id: str, agent_sha256: str) -> bytes:
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode()


def valid_submission_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value))


def _agent_diff(base_agent_py: str, submitted_agent_py: str) -> str:
    return "".join(
        difflib.unified_diff(
            base_agent_py.splitlines(keepends=True),
            submitted_agent_py.splitlines(keepends=True),
            fromfile="a/agent.py",
            tofile="b/agent.py",
        )
    )


def _files_patch(*, base_files: dict[str, str], submitted_files: dict[str, str]) -> str:
    chunks: list[str] = []
    for path in sorted(set(base_files) | set(submitted_files)):
        base_text = base_files.get(path)
        new_text = submitted_files.get(path)
        chunks.append(
            "".join(
                difflib.unified_diff(
                    (base_text or "").splitlines(keepends=True),
                    (new_text or "").splitlines(keepends=True),
                    fromfile=f"a/{path}" if base_text is not None else "/dev/null",
                    tofile=f"b/{path}" if new_text is not None else "/dev/null",
                )
            )
        )
    return "".join(chunks)


def _agent_patch_violations(patch: str) -> list[str]:
    violations: list[str] = []
    current_hunk = ""
    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            current_hunk = raw_line
            continue
        if not raw_line.startswith(("+", "-")) or raw_line.startswith(("+++", "---")):
            continue
        text = raw_line[1:].strip()
        if not text:
            continue
        if any(symbol in current_hunk for symbol in PROTECTED_HUNK_SYMBOLS):
            violations.append(f"agent.py must not edit validator-owned function near `{current_hunk}`.")
        if any(marker in text for marker in PROTECTED_EDIT_MARKERS):
            violations.append(f"agent.py must not edit validator-owned contract line `{text[:100]}`.")
        if not raw_line.startswith("+"):
            continue
        lowered = text.lower()
        for sampling_name in FORBIDDEN_SAMPLING_NAMES:
            if sampling_name in lowered:
                violations.append(f"agent.py must not add miner-controlled sampling parameter `{sampling_name}`.")
        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in lowered:
                violations.append(f"agent.py adds forbidden secret/provider reference `{forbidden}`.")
        if "os.environ" in text or "getenv(" in text:
            env_names = set(re.findall(r"""["']([A-Z][A-Z0-9_]{2,})["']""", text))
            disallowed = sorted(name for name in env_names if name not in ALLOWED_ENV_NAMES)
            if disallowed:
                violations.append(
                    "agent.py reads non-allowlisted environment variable(s): "
                    + ", ".join(disallowed[:8])
                )
    return violations


def _pyflakes_findings(source: str, *, filename: str = "agent.py") -> list[str]:
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except Exception:
        return []

    stdout = io.StringIO()
    stderr = io.StringIO()
    warnings = check(source, filename, Reporter(stdout, stderr))
    if not warnings:
        return []
    findings = [line.strip() for line in (stdout.getvalue() + stderr.getvalue()).splitlines() if line.strip()]
    return [
        f"pyflakes: {line}"
        for line in findings
        if not any(known in line for known in KNOWN_BASELINE_PYFLAKES_SUBSTRINGS)
    ]


def _agent_source_violations(
    source: str,
    *,
    filename: str = "agent.py",
    require_solve: bool = True,
    local_modules: frozenset[str] = frozenset(),
) -> list[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"{filename} must remain valid Python: {exc.msg} at line {exc.lineno}."]

    violations: list[str] = []
    solve = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"), None)
    if solve is None:
        if require_solve:
            violations.append(f"{filename} must define solve(...).")
    else:
        args = [arg.arg for arg in [*solve.args.posonlyargs, *solve.args.args]]
        if tuple(args[: len(REQUIRED_SOLVE_ARGS)]) != REQUIRED_SOLVE_ARGS:
            violations.append(
                "solve() must keep leading arguments: " + ", ".join(REQUIRED_SOLVE_ARGS) + "."
            )
        sampling_args = sorted(name for name in args if name in FORBIDDEN_SAMPLING_NAMES)
        if sampling_args:
            violations.append("solve() must not expose sampling parameter(s): " + ", ".join(sampling_args) + ".")

    stdlib = set(getattr(__import__("sys"), "stdlib_module_names", ()))
    stdlib.update({"__future__"})
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]]
            sampling_args = sorted(name for name in args if name in FORBIDDEN_SAMPLING_NAMES)
            if sampling_args:
                violations.append(
                    f"{node.name}() must not expose sampling parameter(s): "
                    + ", ".join(sampling_args)
                    + "."
                )
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if getattr(key, "value", None) in FORBIDDEN_SAMPLING_NAMES:
                    violations.append(
                        f"{filename} must not set sampling request field `{key.value}`; validator proxy owns sampling."
                    )
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [str(alias.name).split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            roots = [str(node.module or "").split(".", 1)[0]]
        for root in roots:
            if root and root not in stdlib and root not in local_modules:
                violations.append(f"{filename} imports non-stdlib module `{root}`.")
    return violations


def _coerce_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _read_acceptance_ledger(root: Path) -> dict[str, Any]:
    path = root / PRIVATE_SUBMISSION_ACCEPTANCE_LEDGER
    if not path.is_file():
        return {"version": 1, "hotkeys": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"private submission acceptance ledger is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"private submission acceptance ledger is invalid: {path}")
    hotkeys = payload.setdefault("hotkeys", {})
    if not isinstance(hotkeys, dict):
        raise ValueError(f"private submission acceptance ledger has invalid `hotkeys`: {path}")
    payload.setdefault("version", 1)
    return payload


def _write_acceptance_ledger(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / PRIVATE_SUBMISSION_ACCEPTANCE_LEDGER
    with tempfile.NamedTemporaryFile("w", dir=root, encoding="utf-8", delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _read_attempt_ledger(root: Path) -> dict[str, Any]:
    path = root / PRIVATE_SUBMISSION_ATTEMPT_LEDGER
    if not path.is_file():
        return {"version": 1, "hotkeys": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"private submission attempt ledger is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"private submission attempt ledger is invalid: {path}")
    hotkeys = payload.setdefault("hotkeys", {})
    if not isinstance(hotkeys, dict):
        raise ValueError(f"private submission attempt ledger has invalid `hotkeys`: {path}")
    payload.setdefault("version", 1)
    return payload


def _write_attempt_ledger(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / PRIVATE_SUBMISSION_ATTEMPT_LEDGER
    with tempfile.NamedTemporaryFile("w", dir=root, encoding="utf-8", delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _parse_attempted_at(attempt: dict[str, Any]) -> datetime | None:
    value = str(attempt.get("attempted_at") or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
