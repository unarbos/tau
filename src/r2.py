from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import boto3

log = logging.getLogger("swe-eval.r2")

_R2_KEY_PREFIX = "sn66/"
_DASHBOARD_KEY = f"{_R2_KEY_PREFIX}dashboard.json"


def _build_s3_client():
    endpoint = os.environ.get("R2_URL")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    if not all([endpoint, access_key, secret_key]):
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def _get_bucket() -> str:
    return os.environ.get("R2_BUCKET_NAME", "constantinople")


def publish_dashboard_data(
    *,
    current_king: dict[str, Any] | None,
    duel_history: list[dict[str, Any]],
    status: dict[str, Any] | None = None,
) -> bool:
    """Serialize and upload dashboard.json to R2. Returns True on success."""
    client = _build_s3_client()
    if client is None:
        log.warning("R2 credentials not configured; skipping dashboard publish")
        return False

    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "current_king": current_king,
        "duels": duel_history,
        "status": status,
    }
    body = json.dumps(payload, indent=2)
    bucket = _get_bucket()
    try:
        client.put_object(
            Bucket=bucket,
            Key=_DASHBOARD_KEY,
            Body=body.encode(),
            ContentType="application/json",
        )
        log.info("Published dashboard data to r2://%s/%s (%d duels)", bucket, _DASHBOARD_KEY, len(duel_history))
        return True
    except Exception:
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

    return {
        "duel_id": duel_dict.get("duel_id"),
        "started_at": duel_dict.get("started_at"),
        "finished_at": duel_dict.get("finished_at"),
        "king_uid": king_before.get("uid"),
        "king_hotkey": king_before.get("hotkey"),
        "king_repo": king_before.get("repo_full_name"),
        "king_repo_url": f"https://github.com/{king_before.get('repo_full_name', '')}",
        "king_commit_sha": king_before.get("commit_sha"),
        "challenger_uid": challenger.get("uid"),
        "challenger_hotkey": challenger.get("hotkey"),
        "challenger_repo": challenger.get("repo_full_name"),
        "challenger_repo_url": f"https://github.com/{challenger.get('repo_full_name', '')}",
        "challenger_commit_sha": challenger.get("commit_sha"),
        "king_similarity_ratio_mean": (sum(king_ratios) / len(king_ratios)) if king_ratios else 0.0,
        "challenger_similarity_ratio_mean": (sum(challenger_ratios) / len(challenger_ratios)) if challenger_ratios else 0.0,
        "wins": duel_dict.get("wins", 0),
        "losses": duel_dict.get("losses", 0),
        "ties": duel_dict.get("ties", 0),
        "king_replaced": duel_dict.get("king_replaced", False),
        "disqualification_reason": duel_dict.get("disqualification_reason"),
        "rounds": [
            {
                "task_name": r.get("task_name"),
                "winner": r.get("winner"),
                "king_similarity_ratio": r.get("king_similarity_ratio", 0.0),
                "challenger_similarity_ratio": r.get("challenger_similarity_ratio", 0.0),
                "king_challenger_similarity": r.get("king_challenger_similarity", 0.0),
                "king_lines": r.get("king_lines", 0),
                "challenger_lines": r.get("challenger_lines", 0),
            }
            for r in scored_rounds
        ],
    }
