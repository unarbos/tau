from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import httpx

log = logging.getLogger("swe-eval.r2")

_R2_KEY_PREFIX = "sn66/"
_DASHBOARD_KEY = f"{_R2_KEY_PREFIX}dashboard.json"
_DUELS_PREFIX = f"{_R2_KEY_PREFIX}duels/"
_INDEX_KEY = f"{_DUELS_PREFIX}index.json"
_PUBLIC_SENSITIVE_ROUND_BASENAMES = frozenset(
    {
        "commit.json",
        "reference.patch",
        "task.json",
        "task.txt",
    }
)
_PUBLIC_SENSITIVE_SOLVE_RESULT_KEYS = frozenset(
    {
        "raw_output",
        "rollout_format",
        "rollout_filename",
        "session_id",
        "solution_diff",
    }
)
_PUBLIC_SENSITIVE_SOLVE_TOP_LEVEL_KEYS = frozenset(
    {
        "agent_source",
        "commit_sha",
        "repo_full_name",
    }
)

_client_lock = threading.Lock()
_cached_client = None
_client_resolved = False

# Circuit breaker for 429 / SlowDown bursts. When the storage backend
# (Hippius) starts rate-limiting us, retrying immediately just makes the
# throttle worse and burns CPU + log volume. We track the last 429-style
# failure and skip non-essential uploads for _THROTTLE_BACKOFF_SECONDS
# afterwards. Dashboard publishes still go through (they're the user-visible
# heartbeat) but per-artifact uploads back off.
_THROTTLE_LOCK = threading.Lock()
_THROTTLE_UNTIL = 0.0
_THROTTLE_BACKOFF_SECONDS = 60.0
_THROTTLE_LOG_INTERVAL = 30.0
_LAST_THROTTLE_LOG = 0.0
_SUPPRESSED_SINCE_LOG = 0


def _is_throttle_error(exc: BaseException) -> bool:
    """True if the exception looks like S3 rate-limiting (429 / SlowDown)."""
    if isinstance(exc, ClientError):
        meta = exc.response.get("ResponseMetadata", {}) if hasattr(exc, "response") else {}
        if meta.get("HTTPStatusCode") == 429:
            return True
        code = (exc.response.get("Error", {}) or {}).get("Code", "") if hasattr(exc, "response") else ""
        if code in ("SlowDown", "Throttling", "ThrottlingException", "TooManyRequests"):
            return True
    msg = str(exc)
    return "429" in msg or "SlowDown" in msg or "TooManyRequests" in msg


def _note_throttle() -> None:
    """Record that we hit a rate limit; future calls within the backoff
    window will be suppressed. Logs at most once per _THROTTLE_LOG_INTERVAL
    seconds with a count of suppressed uploads."""
    global _THROTTLE_UNTIL, _LAST_THROTTLE_LOG, _SUPPRESSED_SINCE_LOG  # noqa: PLW0603
    now = time.monotonic()
    with _THROTTLE_LOCK:
        _THROTTLE_UNTIL = now + _THROTTLE_BACKOFF_SECONDS
        if now - _LAST_THROTTLE_LOG > _THROTTLE_LOG_INTERVAL:
            log.warning(
                "R2 backend rate-limited (429); backing off %.0fs (suppressed %d uploads since last warning)",
                _THROTTLE_BACKOFF_SECONDS, _SUPPRESSED_SINCE_LOG,
            )
            _LAST_THROTTLE_LOG = now
            _SUPPRESSED_SINCE_LOG = 0


def _is_throttled() -> bool:
    """True if we're currently in the throttle backoff window."""
    global _SUPPRESSED_SINCE_LOG  # noqa: PLW0603
    if time.monotonic() < _THROTTLE_UNTIL:
        with _THROTTLE_LOCK:
            _SUPPRESSED_SINCE_LOG += 1
        return True
    return False


def _get_s3_client():
    """Return a cached boto3 S3 client, or None if credentials are missing."""
    global _cached_client, _client_resolved  # noqa: PLW0603
    if _client_resolved:
        return _cached_client
    with _client_lock:
        if _client_resolved:
            return _cached_client
        endpoint = os.environ.get("R2_URL")
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not all([endpoint, access_key, secret_key]):
            _cached_client = None
        else:
            _cached_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 1, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=30,
                ),
            )
        _client_resolved = True
        return _cached_client


def _get_bucket() -> str:
    return os.environ.get("R2_BUCKET_NAME", "constantinople")


def _upload_json(key: str, data: Any, cache_control: str | None = None) -> bool:
    """Upload a JSON-serializable object to R2. Returns True on success.
    Raises on failure so callers can decide whether to bookkeeping-track
    the failure (e.g. note throttle, log).

    ``cache_control``: optional value for the Cache-Control header on the
    uploaded object. The Hippius edge cache otherwise defaults to
    ``max-age=300, stale-while-revalidate=60`` for application/json which
    makes the public dashboard appear several minutes stale to viewers
    even when we're publishing every few seconds."""
    client = _get_s3_client()
    if client is None:
        return False
    body = json.dumps(data, indent=2)
    extra: dict[str, Any] = {}
    if cache_control:
        extra["CacheControl"] = cache_control
    client.put_object(
        Bucket=_get_bucket(),
        Key=key,
        Body=body.encode(),
        ContentType="application/json",
        **extra,
    )
    return True


def _upload_text(key: str, content: str, content_type: str = "text/plain") -> bool:
    """Upload text content to R2. Returns True on success."""
    client = _get_s3_client()
    if client is None:
        return False
    client.put_object(
        Bucket=_get_bucket(),
        Key=key,
        Body=content.encode(),
        ContentType=content_type,
    )
    return True


def _delete_key(key: str) -> bool:
    """Delete an object from R2. Missing objects are considered successful."""
    client = _get_s3_client()
    if client is None:
        return False
    client.delete_object(Bucket=_get_bucket(), Key=key)
    return True


def _delete_key_quietly(key: str) -> bool:
    try:
        return _delete_key(key)
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.warning("Failed to delete legacy public R2 object %s: %s", key, exc)
        return False


def _delete_keys_batch(keys: list[str]) -> int:
    if not keys:
        return 0
    client = _get_s3_client()
    if client is None:
        return 0
    deleted = 0
    for start in range(0, len(keys), 1000):
        chunk = keys[start:start + 1000]
        try:
            client.delete_objects(
                Bucket=_get_bucket(),
                Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
            )
            deleted += len(chunk)
        except Exception as exc:
            if _is_throttle_error(exc):
                _note_throttle()
                return deleted
            log.warning("Failed to delete %d legacy public R2 objects: %s", len(chunk), exc)
    return deleted


def publish_dashboard_data(
    *,
    current_king: dict[str, Any] | None,
    duel_history: list[dict[str, Any]],
    status: dict[str, Any] | None = None,
) -> bool:
    """Serialize and upload dashboard.json to R2. Returns True on success."""
    if _get_s3_client() is None:
        log.warning("R2 credentials not configured; skipping dashboard publish")
        return False

    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "current_king": current_king,
        "duels": duel_history,
        "status": status,
    }
    try:
        # Short max-age so Hippius's edge cache doesn't make the dashboard
        # look frozen to viewers. We publish every few seconds anyway.
        _upload_json(_DASHBOARD_KEY, payload, cache_control="public, max-age=10")
        log.info("Published dashboard data to r2://%s/%s (%d duels)", _get_bucket(), _DASHBOARD_KEY, len(duel_history))
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish dashboard data to R2")
        return False


def duel_to_summary(duel_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields the dashboard needs from a full DuelResult dict."""
    king_before = duel_dict.get("king_before", {})
    challenger = duel_dict.get("challenger", {})
    rounds = duel_dict.get("rounds", [])

    scored_rounds = [r for r in rounds if r.get("error") is None]
    king_ratios = [r["king_similarity_ratio"] for r in scored_rounds if "king_similarity_ratio" in r]
    challenger_ratios = [r["challenger_similarity_ratio"] for r in scored_rounds if "challenger_similarity_ratio" in r]
    king_scores = [r["king_score"] for r in scored_rounds if "king_score" in r]
    challenger_scores = [r["challenger_score"] for r in scored_rounds if "challenger_score" in r]
    king_llm_scores = [r["king_llm_score"] for r in scored_rounds if "king_llm_score" in r]
    challenger_llm_scores = [r["challenger_llm_score"] for r in scored_rounds if "challenger_llm_score" in r]

    is_confirmation_retest = (
        duel_dict.get("task_set_phase") == "confirmation_retest"
        or duel_dict.get("confirmation_of_duel_id") is not None
    )
    confirmation_retest_passed = duel_dict.get("confirmation_retest_passed")
    if is_confirmation_retest and confirmation_retest_passed is None:
        confirmation_retest_passed = bool(duel_dict.get("king_replaced", False))

    return {
        "duel_id": duel_dict.get("duel_id"),
        "started_at": duel_dict.get("started_at"),
        "finished_at": duel_dict.get("finished_at"),
        "king_uid": king_before.get("uid"),
        "king_hotkey": king_before.get("hotkey"),
        "king_repo": king_before.get("repo_full_name"),
        "king_display_repo_full_name": king_before.get("display_repo_full_name"),
        "king_repo_url": f"https://github.com/{king_before.get('repo_full_name', '')}",
        "king_pr_url": king_before.get("pr_url"),
        "king_commit_sha": king_before.get("commit_sha"),
        "king_display_commit_sha": king_before.get("display_commit_sha"),
        "king_commitment_block": king_before.get("commitment_block"),
        "challenger_uid": challenger.get("uid"),
        "challenger_hotkey": challenger.get("hotkey"),
        "challenger_repo": challenger.get("repo_full_name"),
        "challenger_display_repo_full_name": challenger.get("display_repo_full_name"),
        "challenger_repo_url": f"https://github.com/{challenger.get('repo_full_name', '')}",
        "challenger_pr_url": challenger.get("pr_url"),
        "challenger_commit_sha": challenger.get("commit_sha"),
        "challenger_display_commit_sha": challenger.get("display_commit_sha"),
        "challenger_commitment_block": challenger.get("commitment_block"),
        "king_similarity_ratio_mean": (sum(king_ratios) / len(king_ratios)) if king_ratios else 0.0,
        "challenger_similarity_ratio_mean": (sum(challenger_ratios) / len(challenger_ratios)) if challenger_ratios else 0.0,
        "king_score_mean": (sum(king_scores) / len(king_scores)) if king_scores else 0.0,
        "challenger_score_mean": (sum(challenger_scores) / len(challenger_scores)) if challenger_scores else 0.0,
        "king_llm_score_mean": (sum(king_llm_scores) / len(king_llm_scores)) if king_llm_scores else 0.0,
        "challenger_llm_score_mean": (sum(challenger_llm_scores) / len(challenger_llm_scores)) if challenger_llm_scores else 0.0,
        "wins": duel_dict.get("wins", 0),
        "losses": duel_dict.get("losses", 0),
        "ties": duel_dict.get("ties", 0),
        "errors": duel_dict.get("errors", 0),
        "king_replaced": False if is_confirmation_retest else duel_dict.get("king_replaced", False),
        "disqualification_reason": duel_dict.get("disqualification_reason"),
        "task_set_phase": duel_dict.get("task_set_phase", "primary"),
        "manual_retest_of_duel_id": (
            duel_dict.get("manual_retest_of_duel_id")
            or challenger.get("manual_retest_of_duel_id")
        ),
        "confirmation_of_duel_id": duel_dict.get("confirmation_of_duel_id"),
        "confirmation_duel_id": duel_dict.get("confirmation_duel_id"),
        "confirmation_retest_passed": confirmation_retest_passed,
        "confirmation_failure_reason": duel_dict.get("confirmation_failure_reason"),
        "rounds": [
            {
                "task_name": r.get("task_name"),
                "winner": r.get("winner"),
                "king_similarity_ratio": r.get("king_similarity_ratio", 0.0),
                "challenger_similarity_ratio": r.get("challenger_similarity_ratio", 0.0),
                "king_challenger_similarity": r.get("king_challenger_similarity", 0.0),
                "king_score": r.get("king_score", 0.0),
                "challenger_score": r.get("challenger_score", 0.0),
                "king_llm_score": r.get("king_llm_score", 0.5),
                "challenger_llm_score": r.get("challenger_llm_score", 0.5),
                "llm_judge_winner": r.get("llm_judge_winner", "tie"),
                "llm_judge_rationale": r.get("llm_judge_rationale"),
                "king_lines": r.get("king_lines", 0),
                "challenger_lines": r.get("challenger_lines", 0),
                "baseline_lines": r.get("baseline_lines", 0),
            }
            for r in scored_rounds
        ],
    }


def _duel_key_prefix(duel_id: int) -> str:
    return f"{_DUELS_PREFIX}{duel_id:06d}/"


def _round_key_prefix(duel_id: int, task_name: str) -> str:
    return f"{_duel_key_prefix(duel_id)}rounds/{task_name}/"


def _public_solve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    public_payload = {
        key: value
        for key, value in payload.items()
        if key not in _PUBLIC_SENSITIVE_SOLVE_TOP_LEVEL_KEYS
    }
    result = public_payload.get("result")
    if isinstance(result, dict):
        public_payload["result"] = {
            key: value
            for key, value in result.items()
            if key not in _PUBLIC_SENSITIVE_SOLVE_RESULT_KEYS
        }
    return public_payload


def _public_compare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"commit_sha", "repo_full_name"}
    }


def _public_duel_payload(duel_dict: dict[str, Any]) -> dict[str, Any]:
    public_payload = dict(duel_dict)
    rounds: list[dict[str, Any]] = []
    raw_rounds = duel_dict.get("rounds", [])
    if isinstance(raw_rounds, list):
        for item in raw_rounds:
            if not isinstance(item, dict):
                continue
            rounds.append(
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "challenger_compare_root",
                        "king_compare_root",
                        "task_root",
                    }
                }
            )
    public_payload["rounds"] = rounds
    return public_payload


def _legacy_public_round_leakage_keys(prefix: str) -> list[str]:
    keys = [f"{prefix}{name}" for name in sorted(_PUBLIC_SENSITIVE_ROUND_BASENAMES)]
    keys.append(f"{prefix}solutions/baseline.diff")
    keys.append(f"{prefix}solutions/baseline.solve.json")
    for canonical in ("baseline", "king", "challenger"):
        keys.append(f"{prefix}solutions/{canonical}.rollout.jsonl.gz")
    return keys


def _is_public_task_leakage_key(key: str) -> bool:
    if not key.startswith(_DUELS_PREFIX):
        return False
    basename = key.rsplit("/", 1)[-1]
    if basename == "training.jsonl":
        return True
    if "/rounds/" not in key:
        return False
    if basename in _PUBLIC_SENSITIVE_ROUND_BASENAMES:
        return True
    if "/solutions/" in key and basename == "baseline.diff":
        return True
    if "/solutions/" in key and basename.endswith(".rollout.jsonl.gz"):
        return True
    if "/solutions/" in key and basename == "baseline.solve.json":
        return True
    return False


def purge_public_task_leakage_from_r2(*, prefix: str = _DUELS_PREFIX, dry_run: bool = False) -> int:
    """Delete legacy public objects that reveal private task/reference context."""
    client = _get_s3_client()
    if client is None:
        log.warning("R2 credentials not configured; skipping public leakage purge")
        return 0

    deleted = 0
    pending: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": _get_bucket(), "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = str(item.get("Key") or "")
            if key and _is_public_task_leakage_key(key):
                pending.append(key)
        if len(pending) >= 1000 and not dry_run:
            deleted += _delete_keys_batch(pending)
            pending.clear()
        if not resp.get("IsTruncated"):
            break
        token = str(resp.get("NextContinuationToken") or "")
        if not token:
            break

    if dry_run:
        log.info("Would delete %d legacy public task-leakage object(s) from R2", len(pending))
        return len(pending)
    deleted += _delete_keys_batch(pending)
    log.info("Deleted %d legacy public task-leakage object(s) from R2", deleted)
    return deleted


def publish_round_data(
    *,
    duel_id: int,
    task_name: str,
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Upload public-safe artifacts for a single validation round to R2.

    The local workspace keeps task prompts and reference patches for private
    scoring. Public R2 uploads intentionally exclude task.txt, task.json,
    commit.json, reference.patch, baseline artifacts, model rollouts, and raw
    solve transcripts so miners cannot recover private task/reference context
    from the dashboard API.

    Uploads king/challenger diffs, sanitized king/challenger solve metadata,
    and comparison summaries under:
        sn66/duels/{duel_id}/rounds/{task_name}/...

    ``solution_labels`` maps canonical R2 names to actual on-disk solution
    folder names, e.g. ``{"baseline": "baseline", "challenger": "challenger-42"}``.
    When *None*, falls back to the canonical names as-is.

    Returns True if at least one file was uploaded, False otherwise.
    """
    if _get_s3_client() is None:
        return False
    if _is_throttled():
        return False

    from workspace import build_compare_paths, build_solution_paths, build_task_paths

    prefix = _round_key_prefix(duel_id, task_name)
    task_paths = build_task_paths(tasks_root, task_name)
    labels = solution_labels or {}
    uploaded = 0

    def _handle_upload_exc(exc: BaseException, r2_key: str) -> None:
        if _is_throttle_error(exc):
            _note_throttle()
            return
        log.exception("Failed to upload %s to R2 (non-fatal)", r2_key)

    def _try_upload_public_solve_file(local_path: Path, r2_key: str) -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            data = json.loads(local_path.read_text())
            if not isinstance(data, dict):
                return
            _upload_json(r2_key, _public_solve_payload(data))
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    def _try_upload_public_compare_file(local_path: Path, r2_key: str) -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            data = json.loads(local_path.read_text())
            if not isinstance(data, dict):
                return
            _upload_json(r2_key, _public_compare_payload(data))
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    def _try_upload_text_file(local_path: Path, r2_key: str, content_type: str = "text/plain") -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            _upload_text(r2_key, local_path.read_text(), content_type)
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    for key in _legacy_public_round_leakage_keys(prefix):
        _delete_key_quietly(key)

    canonical_names = ("king", "challenger")
    for canonical in canonical_names:
        disk_name = labels.get(canonical, canonical)
        sol_paths = build_solution_paths(task_paths, disk_name)
        _try_upload_text_file(
            sol_paths.solution_diff_path,
            f"{prefix}solutions/{canonical}.diff",
            "text/x-diff",
        )
        _try_upload_public_solve_file(
            sol_paths.solve_json_path,
            f"{prefix}solutions/{canonical}.solve.json",
        )

    compare_pairs = [
        ("king", "reference"),
        ("challenger", "reference"),
        ("king", "challenger"),
    ]
    for left_canonical, right_canonical in compare_pairs:
        left_disk = labels.get(left_canonical, left_canonical)
        right_default = "baseline" if right_canonical == "reference" else right_canonical
        right_disk = labels.get(right_canonical, right_default)
        disk_cmp_name = f"{left_disk}--vs--{right_disk}"
        r2_cmp_name = f"{left_canonical}--vs--{right_canonical}"
        cmp_paths = build_compare_paths(task_paths, disk_cmp_name)
        _try_upload_public_compare_file(
            cmp_paths.compare_json_path,
            f"{prefix}comparisons/{r2_cmp_name}.json",
        )

    log.info(
        "Published %d round artifacts for duel %d task %s to R2",
        uploaded, duel_id, task_name,
    )
    return uploaded > 0


def publish_duel_data(*, duel_id: int, duel_dict: dict[str, Any]) -> bool:
    """Upload a public-safe DuelResult JSON to R2.

    Writes to: sn66/duels/{duel_id}/duel.json
    """
    if _get_s3_client() is None:
        return False
    if _is_throttled():
        return False
    key = f"{_duel_key_prefix(duel_id)}duel.json"
    try:
        _upload_json(key, _public_duel_payload(duel_dict))
        log.info("Published duel %d to r2://%s/%s", duel_id, _get_bucket(), key)
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish duel %d to R2 (non-fatal)", duel_id)
        return False


def publish_duel_index(
    *,
    duel_history: list[dict[str, Any]],
    latest_duel_dict: dict[str, Any] | None = None,
) -> bool:
    """Rebuild and upload sn66/duels/index.json from the dashboard history.

    Each entry contains enough metadata for discovery plus the list of
    round task names so consumers can construct full key paths.
    """
    if _get_s3_client() is None:
        return False

    public_base_url = os.environ.get("R2_PUBLIC_URL", "")
    entries: list[dict[str, Any]] = []

    round_names_by_duel: dict[int, list[str]] = {}
    if latest_duel_dict:
        did = latest_duel_dict.get("duel_id")
        if did is not None:
            round_names_by_duel[did] = [
                r.get("task_name", "") for r in latest_duel_dict.get("rounds", [])
            ]

    for summary in duel_history:
        duel_id = summary.get("duel_id")
        if duel_id is None:
            continue
        round_task_names = round_names_by_duel.get(
            duel_id,
            [r.get("task_name", "") for r in summary.get("rounds", [])],
        )
        entries.append({
            "duel_id": duel_id,
            "started_at": summary.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "king_uid": summary.get("king_uid"),
            "king_hotkey": summary.get("king_hotkey"),
            "king_repo": summary.get("king_repo"),
            "king_display_repo_full_name": summary.get("king_display_repo_full_name"),
            "king_repo_url": summary.get("king_repo_url"),
            "king_pr_url": summary.get("king_pr_url"),
            "king_commit_sha": summary.get("king_commit_sha"),
            "king_display_commit_sha": summary.get("king_display_commit_sha"),
            "king_commitment_block": summary.get("king_commitment_block"),
            "challenger_uid": summary.get("challenger_uid"),
            "challenger_hotkey": summary.get("challenger_hotkey"),
            "challenger_repo": summary.get("challenger_repo"),
            "challenger_display_repo_full_name": summary.get("challenger_display_repo_full_name"),
            "challenger_repo_url": summary.get("challenger_repo_url"),
            "challenger_pr_url": summary.get("challenger_pr_url"),
            "challenger_commit_sha": summary.get("challenger_commit_sha"),
            "challenger_display_commit_sha": summary.get("challenger_display_commit_sha"),
            "challenger_commitment_block": summary.get("challenger_commitment_block"),
            "king_replaced": summary.get("king_replaced", False),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "ties": summary.get("ties", 0),
            "rounds": round_task_names,
            "path": f"{_DUELS_PREFIX}{duel_id:06d}/",
        })

    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "public_base_url": public_base_url,
        "duels": entries,
    }
    if _is_throttled():
        return False
    try:
        _upload_json(_INDEX_KEY, payload, cache_control="public, max-age=30")
        log.info("Published duel index (%d entries) to R2", len(entries))
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish duel index to R2 (non-fatal)")
        return False


def backfill_duel_to_r2(
    duel_json_path: Path,
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Upload a historical duel and its round artifacts to R2.

    Reads the full duel JSON from disk, uploads the duel record, then
    iterates over rounds and uploads each round's artifacts if available.
    Returns True if the duel record was uploaded.
    """
    if _get_s3_client() is None:
        log.warning("R2 credentials not configured; skipping backfill")
        return False

    duel_dict = json.loads(duel_json_path.read_text())
    duel_id = duel_dict["duel_id"]

    if not publish_duel_data(duel_id=duel_id, duel_dict=duel_dict):
        return False

    for round_data in duel_dict.get("rounds", []):
        task_name = round_data.get("task_name")
        if not task_name:
            continue
        try:
            publish_round_data(
                duel_id=duel_id, task_name=task_name,
                tasks_root=tasks_root, solution_labels=solution_labels,
            )
        except Exception:
            log.exception(
                "Backfill: failed to upload round %s for duel %d (non-fatal)",
                task_name, duel_id,
            )

    log.info("Backfilled duel %d from %s", duel_id, duel_json_path)
    return True


def publish_training_data(
    *,
    duel_id: int,
    duel_dict: dict[str, Any],
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Remove legacy public training data.

    The historical training.jsonl format was self-contained, which meant it
    exposed private task prompts and reference diffs on public R2. Training
    exports need a private destination; public R2 should only carry sanitized
    dashboard/duel artifacts.
    """
    del duel_dict, tasks_root, solution_labels
    if _get_s3_client() is None:
        return False
    if _is_throttled():
        return False
    key = f"{_duel_key_prefix(duel_id)}training.jsonl"
    try:
        deleted = _delete_key(key)
        if deleted:
            log.info("Removed public training data for duel %d from R2", duel_id)
        return False
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to remove public training data for duel %d (non-fatal)", duel_id)
        return False


def fetch_chain_data(netuid: int) -> dict[str, Any] | None:
    """Fetch subnet and market data from the TaoMarketCap API."""
    api_key = os.environ.get("TMC_API_KEY")
    if not api_key:
        return None
    headers = {"Authorization": api_key, "Accept": "application/json"}
    base = "https://api.taomarketcap.com/public/v1"
    try:
        with httpx.Client(timeout=15, headers=headers) as c:
            market = c.get(f"{base}/market/market-data/")
            subnet = c.get(f"{base}/subnets/{netuid}/")
            weights = c.get(f"{base}/subnets/weights/{netuid}/")
        m = market.json() if market.status_code == 200 else {}
        s = subnet.json() if subnet.status_code == 200 else {}
        w = weights.json() if weights.status_code == 200 else {}
        snap = s.get("latest_snapshot", {})
        burn = int(snap.get("burn", 0))
        tao = float(m.get("current_price", 0))
        alpha_tao = float(snap.get("subnet_moving_price", 0))
        wt = []
        for we in w.get("weights", []):
            for tid, val in we.get("value", {}).items():
                wt.append({"validator_uid": we["uid"], "miner_uid": int(tid), "weight": val})
        return {
            "fetched_at": datetime.now(tz=UTC).isoformat(),
            "tao_price_usd": tao,
            "tao_change_24h": float((m.get("usd_quote") or {}).get("percent_change_24h", 0)),
            "tao_market_cap": float((m.get("usd_quote") or {}).get("market_cap", 0)),
            "alpha_price_tao": alpha_tao,
            "alpha_price_usd": alpha_tao * tao,
            "subnet_tao": int(snap.get("subnet_tao", 0)) / 1e9,
            "subnet_emission_per_day": int(snap.get("subnet_tao_in_emission", 0)) / 1e9 * 7200,
            "burn_cost_rao": burn,
            "burn_cost_tao": burn / 1e9,
            "burn_cost_usd": burn / 1e9 * tao,
            "neuron_count": int(snap.get("subnetwork_n", 0)),
            "max_neurons": int(snap.get("max_allowed_uids", 256)),
            "token_symbol": snap.get("token_symbol", ""),
            "subnet_name": (snap.get("subnet_identities_v3") or {}).get("subnetName", ""),
            "tempo": int(snap.get("tempo", 0)),
            "immunity_period": int(snap.get("immunity_period", 0)),
            "weights": wt,
        }
    except Exception:
        log.exception("Failed to fetch chain data (non-fatal)")
        return None
