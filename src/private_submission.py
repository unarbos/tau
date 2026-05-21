from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import py_compile
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

DEFAULT_OPENROUTER_MIN_SCORE = 70
PRIVATE_SUBMISSION_ACCEPTANCE_LEDGER = "_accepted_submissions.json"
MINER_HOTKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,64}$")
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


def run_private_submission_checks(
    *,
    hotkey: str,
    submitted_agent_py: str,
    base_agent_py: str,
    openrouter_judge: JudgeFn | None = None,
    min_score: int = DEFAULT_OPENROUTER_MIN_SCORE,
) -> PrivateSubmissionCheckResult:
    agent_sha = hashlib.sha256(submitted_agent_py.encode("utf-8")).hexdigest()
    patch = _agent_diff(base_agent_py, submitted_agent_py)
    smoke = run_agent_smoke_checks(agent_py=submitted_agent_py)
    scope_guard = run_scope_guard(hotkey=hotkey, agent_py=submitted_agent_py, patch=patch)
    checks = {"agent_smoke": smoke, "scope_guard": scope_guard}
    if smoke.status == "passed" and scope_guard.status == "passed":
        checks["openrouter_judge"] = run_openrouter_judge_gate(
            hotkey=hotkey,
            base_agent_py=base_agent_py,
            submitted_agent_py=submitted_agent_py,
            patch=patch,
            judge=openrouter_judge,
            min_score=min_score,
        )
    else:
        failed = "agent smoke" if smoke.status != "passed" else "scope guard"
        checks["openrouter_judge"] = SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="skipped",
            summary=f"Skipped because {failed} failed.",
        )
    accepted = all(check.status in {"passed", "warn"} for check in checks.values())
    return PrivateSubmissionCheckResult(accepted=accepted, agent_sha256=agent_sha, checks=checks)


def run_agent_smoke_checks(*, agent_py: str) -> SubmissionCheck:
    findings: list[str] = []
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as tmp:
            tmp.write(agent_py)
            tmp_path = Path(tmp.name)
        try:
            py_compile.compile(str(tmp_path), doraise=True)
        finally:
            tmp_path.unlink(missing_ok=True)
    except py_compile.PyCompileError as exc:
        findings.append(f"agent.py failed py_compile: {exc.msg}")
    except OSError as exc:
        findings.append(f"agent.py smoke check could not create temp file: {exc}")

    pyflakes_findings = _pyflakes_findings(agent_py)
    findings.extend(pyflakes_findings)
    findings = _dedupe(findings)
    if findings:
        return SubmissionCheck(
            name="Agent Smoke",
            status="failed",
            summary="agent.py failed local smoke checks.",
            findings=findings,
        )
    return SubmissionCheck(
        name="Agent Smoke",
        status="passed",
        summary="agent.py compiles and has no new pyflakes findings.",
    )


def run_scope_guard(*, hotkey: str, agent_py: str, patch: str) -> SubmissionCheck:
    findings: list[str] = []
    findings.extend(_agent_patch_violations(patch))
    findings.extend(_agent_source_violations(agent_py))
    findings = _dedupe(findings)
    if findings:
        return SubmissionCheck(
            name="Submission Scope Guard",
            status="failed",
            summary="agent.py failed the local submission contract checks.",
            findings=findings,
        )
    return SubmissionCheck(
        name="Submission Scope Guard",
        status="passed",
        summary="agent.py satisfies the local submission contract checks.",
    )


def run_openrouter_judge_gate(
    *,
    hotkey: str,
    base_agent_py: str,
    submitted_agent_py: str,
    patch: str,
    judge: JudgeFn | None,
    min_score: int,
) -> SubmissionCheck:
    if judge is None:
        return SubmissionCheck(
            name="OpenRouter Submission Judge",
            status="skipped",
            summary="No local OpenRouter judge function was configured.",
        )
    payload = {
        "hotkey": hotkey,
        "title": f"{hotkey} private submission",
        "changed_files": [{"filename": "agent.py", "status": "modified"}],
        "static_findings": {
            "fail_reasons": [],
            "warnings": [],
            "findings": [],
            "changed_files": ["agent.py"],
        },
        "patch": patch,
        "base_agent_py": base_agent_py,
        "submitted_agent_py": submitted_agent_py,
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

    component_floor = max(0, min(100, int(min_score) - 5))
    for name in ("real_edit_score", "safety_score", "scope_score", "contract_score"):
        if name not in judgment:
            continue
        score = _coerce_score(judgment.get(name))
        if score < component_floor:
            failures.append(f"{name}={score} is below component minimum {component_floor}.")

    if "real_edit_score" in judgment:
        real_edit = _coerce_score(judgment.get("real_edit_score"))
        if overall > real_edit + 10:
            failures.append(
                f"overall_score={overall} exceeds real_edit_score={real_edit} by more than 10 points."
            )
    return failures


def write_private_submission_bundle(
    *,
    root: Path,
    submission_id: str,
    hotkey: str,
    agent_py: str,
    check_result: PrivateSubmissionCheckResult,
    signature: str,
    registration_block: int | None = None,
    agent_username: str | None = None,
    coldkey: str | None = None,
    coldkey_signature: str | None = None,
    overwrite: bool = False,
) -> Path:
    if not valid_submission_id(submission_id):
        raise ValueError("submission_id must contain only letters, numbers, '.', '_' or '-'")
    target = root / submission_id
    if target.exists() and not overwrite:
        raise FileExistsError(f"private submission already exists: {submission_id}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "agent.py").write_text(agent_py, encoding="utf-8")
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
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ledger = _read_acceptance_ledger(root)
    ledger.setdefault("version", 1)
    hotkeys = ledger.setdefault("hotkeys", {})
    if not isinstance(hotkeys, dict):
        raise ValueError("private submission acceptance ledger has invalid `hotkeys`")
    hotkeys[hotkey] = {
        "registration_block": int(registration_block),
        "submission_id": submission_id,
        "agent_sha256": agent_sha256.lower(),
        "accepted_at": datetime.now(UTC).isoformat(),
        **_verified_identity_metadata(
            agent_username=agent_username,
            coldkey=coldkey,
            coldkey_signature=coldkey_signature,
        ),
    }
    _write_acceptance_ledger(root, ledger)


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
    actual_sha = hashlib.sha256(agent_path.read_bytes()).hexdigest()
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
    return f"tau-private-submission-v1:{hotkey}:{submission_id}:{agent_sha256.lower()}".encode("utf-8")


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


def _pyflakes_findings(source: str) -> list[str]:
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except Exception:
        return []

    stdout = io.StringIO()
    stderr = io.StringIO()
    warnings = check(source, "agent.py", Reporter(stdout, stderr))
    if not warnings:
        return []
    findings = [line.strip() for line in (stdout.getvalue() + stderr.getvalue()).splitlines() if line.strip()]
    return [
        f"pyflakes: {line}"
        for line in findings
        if not any(known in line for known in KNOWN_BASELINE_PYFLAKES_SUBSTRINGS)
    ]


def _agent_source_violations(source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename="agent.py")
    except SyntaxError as exc:
        return [f"agent.py must remain valid Python: {exc.msg} at line {exc.lineno}."]

    violations: list[str] = []
    solve = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"), None)
    if solve is None:
        violations.append("agent.py must define solve(...).")
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
                        f"agent.py must not set sampling request field `{key.value}`; validator proxy owns sampling."
                    )
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [str(alias.name).split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [str(node.module or "").split(".", 1)[0]]
        for root in roots:
            if root and root not in stdlib:
                violations.append(f"agent.py imports non-stdlib module `{root}`.")
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


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
