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
import httpx
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from tau.io.r2 import BotoS3Client, LocalS3Client, S3Client

log = logging.getLogger("swe-eval.r2")

_R2_KEY_PREFIX = "sn66/"
_DASHBOARD_KEY = f"{_R2_KEY_PREFIX}dashboard.json"
_DASHBOARD_HOME_KEY = f"{_R2_KEY_PREFIX}dashboard-home.json"
_SUBMISSIONS_API_KEY = f"{_R2_KEY_PREFIX}api/submissions"
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
_cached_client: S3Client | None = None
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


def _get_s3_client() -> S3Client | None:
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
            _cached_client = BotoS3Client(boto3.client(
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
            ))
        _client_resolved = True
        return _cached_client



def _get_client() -> S3Client | None:
    local_path = os.environ.get("R2_LOCAL_PATH")
    if local_path:
        return LocalS3Client(Path(local_path))
    return _get_s3_client()


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
    client = _get_client()
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
    client = _get_client()
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
    client = _get_client()
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
    client = _get_client()
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


def build_dashboard_payload(
    *,
    current_king: dict[str, Any] | None,
    duel_history: list[dict[str, Any]],
    status: dict[str, Any] | None = None,
    benchmarks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "current_king": current_king,
        "duels": duel_history,
        "status": status,
        "benchmarks": benchmarks or {},
    }


def build_dashboard_home_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _dashboard_home_payload(payload)


def _copy_fields(source: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    return {field: source.get(field) for field in fields if field in source}


def _round_count(source: dict[str, Any] | None) -> int:
    if not isinstance(source, dict):
        return 0
    count = source.get("round_count")
    if isinstance(count, int) and count >= 0:
        return count
    rounds = source.get("rounds")
    return len(rounds) if isinstance(rounds, list) else 0


def _dashboard_submission_summary(source: dict[str, Any] | None) -> dict[str, Any] | None:
    return _copy_fields(source, (
        "uid", "hotkey", "agent_username", "coldkey", "repo", "repo_full_name",
        "repo_url", "pr_url", "pr_number", "commit_sha", "display_repo_full_name",
        "display_repo_url", "display_commit_sha", "runtime_commit_sha",
        "runtime_repo_full_name", "runtime_repo_url", "source", "share", "king_since",
        "king_duels_defended", "hold_seconds", "accepted_at", "commitment",
        "accepted_at", "base_repo_full_name",
    ))


def _dashboard_round_summary(source: dict[str, Any] | None) -> dict[str, Any]:
    summary = _copy_fields(source, ("task_name", "winner", "llm_judge_winner", "error", "task_error")) or {}
    if _is_task_error_round_dict(source):
        summary["winner"] = "tie"
        summary["llm_judge_winner"] = summary.get("llm_judge_winner") or "tie"
        summary["task_error"] = summary.get("task_error") or summary.get("error")
    return summary


def _is_task_error_round_dict(round_dict: dict[str, Any] | None) -> bool:
    if not isinstance(round_dict, dict):
        return False
    if round_dict.get("task_error"):
        return True
    error = str(round_dict.get("error") or "")
    return error.startswith("task_error:")


_PUBLIC_JUDGE_RATIONALE_WITHHELD = "Detailed judge rationale withheld from public dashboard."


def public_judge_rationale(
    *,
    rationale: str | None,
    llm_judge_winner: str | None,
) -> str | None:
    """Return a censored judge note for public R2/dashboard payloads."""
    if not rationale:
        return None
    winner = str(llm_judge_winner or "tie").strip().lower()
    if winner not in {"king", "challenger", "tie"}:
        winner = "tie"
    return f"LLM judge verdict: {winner.upper()}. {_PUBLIC_JUDGE_RATIONALE_WITHHELD}"


def _public_round_judge_rationale(round_dict: dict[str, Any]) -> str | None:
    return public_judge_rationale(
        rationale=round_dict.get("llm_judge_rationale"),
        llm_judge_winner=round_dict.get("llm_judge_winner") or round_dict.get("winner"),
    )


def _sanitize_public_round_dict(round_dict: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(round_dict)
    censored = _public_round_judge_rationale(round_dict)
    if censored is not None:
        sanitized["llm_judge_rationale"] = censored
    elif "llm_judge_rationale" in sanitized:
        sanitized.pop("llm_judge_rationale")
    return sanitized


def _dashboard_duel_summary(source: dict[str, Any]) -> dict[str, Any]:
    summary = _copy_fields(source, (
        "duel_id", "started_at", "finished_at", "king_replaced", "disqualification_reason",
        "confirmation_duel_id", "confirmation_retest_passed", "confirmation_failure_reason",
        "confirmation_of_duel_id", "manual_retest_of_duel_id", "wins", "losses", "ties",
        "pause_reason", "status_message",
        "errors", "threshold", "duel_rounds", "task_set_phase", "king_uid", "king_hotkey",
        "king_agent_username", "king_repo", "king_repo_url", "king_pr_url", "king_commit_sha",
        "king_commitment_block", "challenger_uid", "challenger_hotkey",
        "challenger_agent_username", "hotkey", "challenger_repo", "challenger_repo_url",
        "challenger_pr_url", "challenger_commit_sha", "challenger_commitment_block",
    )) or {}
    summary["round_count"] = _round_count(source)
    return summary


def _dashboard_active_duel_summary(source: dict[str, Any] | None) -> dict[str, Any] | None:
    summary = _copy_fields(source, (
        "duel_id", "phase", "status", "challenger_uid", "challenger_hotkey",
        "challenger_agent_username", "challenger_repo", "challenger_repo_url", "challenger_pr_url",
        "king_uid", "king_hotkey", "king_agent_username", "king_repo", "king_repo_url",
        "king_pr_url", "duel_rounds", "target_round_count", "gathered_tasks", "needed_tasks",
        "wins", "losses", "ties", "threshold", "task_set_phase", "confirmation_of_duel_id",
        "confirmation_duel_id", "manual_retest_of_duel_id", "pool_size", "pause_reason", "status_message",
        "published_round_count",
    ))
    if summary is None:
        return None
    rounds = source.get("rounds") if isinstance(source, dict) else None
    summary["rounds"] = [_dashboard_round_summary(item) for item in rounds] if isinstance(rounds, list) else []
    return summary


def _dashboard_status_summary(source: dict[str, Any] | None, duels: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    source = _status_with_corrected_recent_king_counts(source, duels or [])
    if not isinstance(source, dict):
        return {}
    summary = _copy_fields(source, (
        "netuid", "total_rounds", "miners_seen", "king_duels_defended", "king_since",
        "validator_started_at", "scoring", "links",
    )) or {}
    for key in ("recent_kings", "queue", "disqualified", "retired"):
        summary[key] = [_dashboard_submission_summary(item) for item in source.get(key, []) if isinstance(item, dict)]
    summary["active_duel"] = _dashboard_active_duel_summary(source.get("active_duel"))
    return summary


def build_dashboard_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    duels = payload.get("duels")

    _links = payload.get("links")
    links = _links if isinstance(_links, dict) else {}
    status = _dashboard_status_summary(payload.get("status"), duels if isinstance(duels, list) else [])
    return {
        "updated_at": payload.get("updated_at"),
        "current_king": _dashboard_submission_summary(payload.get("current_king")),
        "duels": [_dashboard_duel_summary(item) for item in duels if isinstance(item, dict)] if isinstance(duels, list) else [],
        "duels_total": len(duels) if isinstance(duels, list) else 0,
        "status": status,
        "active_duel": status.get("active_duel"),
        "benchmarks": payload.get("benchmarks") if isinstance(payload.get("benchmarks"), dict) else {},
        "links": {**links, "duels_html": "./duels.html", "dashboard_full": "./dashboard-summary.json"},
    }



def _same_dashboard_participant(left: dict[str, Any] | None, right: dict[str, Any] | None, prefix: str = "") -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    uid = left.get(f"{prefix}_uid") if prefix else left.get("uid")
    hotkey = left.get(f"{prefix}_hotkey") if prefix else left.get("hotkey")
    commit = (
        left.get(f"{prefix}_display_commit_sha")
        or left.get(f"{prefix}_commit_sha")
        if prefix
        else left.get("display_commit_sha") or left.get("commit_sha")
    )
    right_uid = right.get("uid") or right.get("king_uid") or right.get("challenger_uid")
    right_hotkey = right.get("hotkey") or right.get("king_hotkey") or right.get("challenger_hotkey")
    right_commits = [
        str(value).lower()
        for value in (right.get("display_commit_sha"), right.get("commit_sha"))
        if value
    ]
    if uid is not None and right_uid is not None and str(uid) != str(right_uid):
        return False
    if hotkey and right_hotkey and str(hotkey) != str(right_hotkey):
        return False
    if commit and right_commits and not any(value.startswith(str(commit).lower()) or str(commit).lower().startswith(value) for value in right_commits):
        return False
    return any(value is not None and value != "" for value in (uid, hotkey, commit))


def _duel_id(source: dict[str, Any]) -> int:
    try:
        return int(source.get("duel_id") or 0)
    except (TypeError, ValueError):
        return 0


def _linked_confirmation_failed(duel: dict[str, Any], duels_by_id: dict[int, dict[str, Any]]) -> bool:
    if duel.get("confirmation_retest_passed") is False:
        return True
    try:
        confirmation_id = int(duel.get("confirmation_duel_id") or 0)
    except (TypeError, ValueError):
        confirmation_id = 0
    linked = duels_by_id.get(confirmation_id) if confirmation_id else None
    return isinstance(linked, dict) and linked.get("confirmation_retest_passed") is False


def _real_king_transition(duel: dict[str, Any], duels_by_id: dict[int, dict[str, Any]]) -> bool:
    if duel.get("king_replaced") is not True:
        return False
    if duel.get("task_set_phase") == "confirmation_retest" or duel.get("confirmation_of_duel_id") is not None:
        return False
    if duel.get("disqualification_reason"):
        return False
    return not _linked_confirmation_failed(duel, duels_by_id)


def _transition_matches(duel: dict[str, Any], participant: dict[str, Any], prefix: str, duels_by_id: dict[int, dict[str, Any]]) -> bool:
    return _real_king_transition(duel, duels_by_id) and _same_dashboard_participant(duel, participant, prefix)


def _dashboard_defense_count(participant: dict[str, Any], duels: list[dict[str, Any]]) -> int:
    ordered = sorted((duel for duel in duels if isinstance(duel, dict)), key=_duel_id)
    duels_by_id = {_duel_id(duel): duel for duel in ordered if _duel_id(duel)}
    start_id = 0
    for duel in ordered:
        if _transition_matches(duel, participant, "challenger", duels_by_id):
            start_id = _duel_id(duel)
    end_id = None
    for duel in ordered:
        duel_id = _duel_id(duel)
        if duel_id <= start_id:
            continue
        if _transition_matches(duel, participant, "king", duels_by_id):
            end_id = duel_id
            break

    count = 0
    for duel in ordered:
        duel_id = _duel_id(duel)
        if duel_id <= start_id:
            continue
        if end_id is not None and duel_id >= end_id:
            continue
        if not _same_dashboard_participant(duel, participant, "king"):
            continue
        if duel.get("task_set_phase") == "confirmation_retest" or duel.get("confirmation_of_duel_id") is not None:
            continue
        if duel.get("disqualification_reason"):
            continue
        if _real_king_transition(duel, duels_by_id):
            continue
        count += 1
    return count


def _dashboard_queue_item(source: Any) -> Any:
    if not isinstance(source, dict):
        return source
    if not str(source.get("source") or "").startswith("private"):
        return source
    return {
        key: value
        for key, value in source.items()
        if key not in {"submission_block", "registration_block", "commitment_block"}
    }


def _status_with_corrected_recent_king_counts(status: Any, duels: list[dict[str, Any]]) -> Any:
    if not isinstance(status, dict):
        return status
    recent = status.get("recent_kings")
    corrected_recent = recent
    if isinstance(recent, list):
        corrected_recent = []
        for item in recent:
            if not isinstance(item, dict):
                corrected_recent.append(item)
                continue
            corrected_recent.append({**item, "king_duels_defended": _dashboard_defense_count(item, duels)})
    queue = status.get("queue")
    corrected_queue = [_dashboard_queue_item(item) for item in queue] if isinstance(queue, list) else queue
    return {**status, "recent_kings": corrected_recent, "queue": corrected_queue}

def publish_dashboard_data(
    *,
    current_king: dict[str, Any] | None,
    duel_history: list[dict[str, Any]],
    status: dict[str, Any] | None = None,
    benchmarks: dict[str, Any] | None = None,
) -> bool:
    """Serialize and upload dashboard home/summary data to R2. Returns True on success."""
    if _get_client() is None:
        log.warning("R2 credentials not configured; skipping dashboard publish")
        return False

    payload = build_dashboard_payload(
        current_king=current_king,
        duel_history=duel_history,
        status=status,
        benchmarks=benchmarks,
    )
    home_payload = build_dashboard_home_payload(payload)
    try:
        # Short max-age so Hippius's edge cache doesn't make the dashboard
        # look frozen to viewers. We publish every few seconds anyway.
        summary_payload = build_dashboard_summary_payload(payload)
        _upload_json(_DASHBOARD_HOME_KEY, home_payload, cache_control="public, max-age=10")
        _upload_json(f"{_R2_KEY_PREFIX}dashboard-summary.json", summary_payload, cache_control="public, max-age=10")
        log.info(
            "Published dashboard home/summary to r2://%s/%s (%d duels)",
            _get_bucket(),
            _DASHBOARD_HOME_KEY,
            len(duel_history),
        )
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish dashboard data to R2")
        return False


def publish_benchmark_data(*, benchmark_payload: dict[str, Any]) -> bool:
    """Merge benchmark summary data into the public dashboard objects."""
    if _get_client() is None:
        log.warning("R2 credentials not configured; skipping benchmark publish")
        return False
    try:
        dashboard = _download_dashboard_payload()
        _benchmarks = dashboard.get("benchmarks")
        benchmarks = _benchmarks if isinstance(_benchmarks, dict) else {}
        benchmarks.update(benchmark_payload)
        dashboard["benchmarks"] = benchmarks
        dashboard["updated_at"] = datetime.now(tz=UTC).isoformat()
        home_payload = build_dashboard_home_payload(dashboard)
        summary_payload = build_dashboard_summary_payload(dashboard)
        _upload_json(_DASHBOARD_KEY, dashboard, cache_control="public, max-age=10")
        _upload_json(_DASHBOARD_HOME_KEY, home_payload, cache_control="public, max-age=10")
        _upload_json(f"{_R2_KEY_PREFIX}dashboard-summary.json", summary_payload, cache_control="public, max-age=10")
        _upload_json(
            f"{_R2_KEY_PREFIX}swebench-local.json",
            _public_swebench_payload(benchmarks),
            cache_control="public, max-age=10",
        )
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish benchmark data to R2")
        return False


def _public_swebench_payload(benchmarks: dict[str, Any]) -> dict[str, Any]:
    swebench = benchmarks.get("swebench_verified") if isinstance(benchmarks, dict) else {}
    latest = swebench.get("latest") if isinstance(swebench, dict) and isinstance(swebench.get("latest"), dict) else None
    normalized = _normalize_public_swebench_latest(latest)
    return {"latest": normalized, "active": normalized}


def _normalize_public_swebench_latest(latest: dict[str, Any] | None) -> dict[str, Any] | None:
    if latest is None or isinstance(latest.get("scores"), dict):
        return latest
    return {
        **latest,
        "scores": {
            "king": latest.get("king"),
            "baseline": latest.get("baseline") or latest.get("pi"),
            "pi": latest.get("pi"),
            "delta_pass_rate": latest.get("delta_pass_rate"),
        },
    }


def _download_dashboard_payload() -> dict[str, Any]:
    client = _get_client()
    if client is None:
        return {}
    try:
        response = client.get_object(Bucket=_get_bucket(), Key=_DASHBOARD_KEY)
        body = response["Body"].read()
        payload = json.loads(body.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _dashboard_home_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the small first-paint dashboard payload.

    Full history stays in dashboard.json and duels/index.json. The home payload
    keeps only the most recent duel window so the landing page can update often
    without downloading megabytes of old rounds on every poll.
    """
    duels = payload.get("duels")
    recent_duels = [_dashboard_home_duel(item) for item in duels[-40:]] if isinstance(duels, list) else []
    _links = payload.get("links")
    links = _links if isinstance(_links, dict) else {}
    return {
        "updated_at": payload.get("updated_at"),
        "current_king": payload.get("current_king"),
        "duels": recent_duels,
        "duels_total": len(duels) if isinstance(duels, list) else 0,
        "status": _status_with_corrected_recent_king_counts(payload.get("status"), duels if isinstance(duels, list) else []),
        "links": {
            **links,
            "dashboard_full": "./dashboard.json",
            "duels_index": "./duels/index.json",
        },
    }


def _dashboard_home_duel(duel: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "duel_id",
        "started_at",
        "finished_at",
        "king_replaced",
        "disqualification_reason",
        "confirmation_duel_id",
        "confirmation_retest_passed",
        "confirmation_failure_reason",
        "confirmation_of_duel_id",
        "manual_retest_of_duel_id",
        "wins",
        "losses",
        "ties",
        "errors",
        "threshold",
        "duel_rounds",
        "task_set_phase",
        "king_uid",
        "king_hotkey",
        "king_agent_username",
        "king_repo",
        "king_repo_url",
        "king_pr_url",
        "king_commit_sha",
        "king_commitment_block",
        "challenger_uid",
        "challenger_hotkey",
        "challenger_agent_username",
        "challenger_repo",
        "challenger_repo_url",
        "challenger_pr_url",
        "challenger_commit_sha",
        "challenger_commitment_block",
    )
    summary = {field: duel.get(field) for field in fields if field in duel}
    rounds = duel.get("rounds")
    summary["round_count"] = len(rounds) if isinstance(rounds, list) else int(duel.get("round_count") or 0)
    return summary


def publish_submissions_api_data(payload: dict[str, Any]) -> bool:
    """Upload the public private-submissions API payload to R2."""
    if _get_client() is None:
        log.warning("R2 credentials not configured; skipping submissions API publish")
        return False

    try:
        _upload_json(_SUBMISSIONS_API_KEY, payload, cache_control="public, max-age=10")
        log.info(
            "Published submissions API data to r2://%s/%s (%d submissions)",
            _get_bucket(),
            _SUBMISSIONS_API_KEY,
            len(payload.get("submissions", [])),
        )
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish submissions API data to R2")
        return False


def duel_to_summary(duel_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields the dashboard needs from a full DuelResult dict."""
    king_before = duel_dict.get("king_before", {})
    challenger = duel_dict.get("challenger", {})
    rounds = duel_dict.get("rounds", [])

    scored_rounds = [r for r in rounds if r.get("error") is None or _is_task_error_round_dict(r)]
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
        "errors": sum(
            1
            for r in rounds
            if r.get("winner") == "error" and not _is_task_error_round_dict(r)
        ),
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
                "winner": "tie" if _is_task_error_round_dict(r) else r.get("winner"),
                "king_similarity_ratio": r.get("king_similarity_ratio", 0.0),
                "challenger_similarity_ratio": r.get("challenger_similarity_ratio", 0.0),
                "king_challenger_similarity": r.get("king_challenger_similarity", 0.0),
                "king_score": r.get("king_score", 0.0),
                "challenger_score": r.get("challenger_score", 0.0),
                "king_llm_score": r.get("king_llm_score", 0.5),
                "challenger_llm_score": r.get("challenger_llm_score", 0.5),
                "llm_judge_winner": (
                    "tie" if _is_task_error_round_dict(r) else r.get("llm_judge_winner", "tie")
                ),
                "llm_judge_rationale": _public_round_judge_rationale(r),
                "task_error": (
                    r.get("task_error") or r.get("error")
                    if _is_task_error_round_dict(r)
                    else r.get("task_error")
                ),
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
            round_payload = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "challenger_compare_root",
                    "king_compare_root",
                    "task_root",
                }
            }
            rounds.append(_sanitize_public_round_dict(round_payload))
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
    client = _get_client()
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
    commit.json, reference.patch, legacy baseline artifacts, model rollouts, and raw
    solve transcripts so miners cannot recover private task/reference context
    from the dashboard API.

    Uploads king/challenger diffs, sanitized king/challenger solve metadata,
    and comparison summaries under:
        sn66/duels/{duel_id}/rounds/{task_name}/...

    ``solution_labels`` maps canonical R2 names to actual on-disk solution
    folder names, e.g. ``{"reference": "reference", "challenger": "challenger-42"}``.
    When *None*, falls back to the canonical names as-is.

    Returns True if at least one file was uploaded, False otherwise.
    """
    if _get_client() is None:
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
        right_disk = labels.get(right_canonical, right_canonical)
        disk_cmp_name = f"{left_disk}--vs--{right_disk}"
        if right_canonical == "reference":
            candidate_paths = build_compare_paths(task_paths, disk_cmp_name)
            if not candidate_paths.compare_json_path.exists():
                disk_cmp_name = f"{left_disk}--vs--baseline"
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
    if _get_client() is None:
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
    if _get_client() is None:
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
            "disqualification_reason": summary.get("disqualification_reason"),
            "confirmation_duel_id": summary.get("confirmation_duel_id"),
            "confirmation_of_duel_id": summary.get("confirmation_of_duel_id"),
            "confirmation_retest_passed": summary.get("confirmation_retest_passed"),
            "confirmation_failure_reason": summary.get("confirmation_failure_reason"),
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
    if _get_client() is None:
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
    if _get_client() is None:
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
