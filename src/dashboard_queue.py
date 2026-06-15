"""Keep dashboard queue snapshots fresh from the private-submission ledger.

The validator is the authoritative queue processor, but the dashboard must not
freeze when validator is stopped. serve.py calls these helpers to overlay any
accepted submissions that are not already reflected in the published snapshot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from private_submission import accepted_private_submission_entries

try:
    from hotkey_uid_cache import hotkey_uid_map
except Exception:  # pragma: no cover - optional during lightweight imports
    hotkey_uid_map = None  # type: ignore[assignment]

_PRIVATE_SOURCE = "private-submission-api"


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ledger_root(validate_root: Path) -> Path:
    return validate_root / "private-submissions"


def _sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    accepted_at = str(item.get("accepted_at") or "")
    if accepted_at:
        return (0, accepted_at)
    block = item.get("commitment_block")
    try:
        return (1, int(block))
    except (TypeError, ValueError):
        return (2, str(item.get("hotkey") or ""))


def _queue_item_from_ledger_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    hotkey = str(entry.get("hotkey") or "").strip()
    submission_id = str(entry.get("submission_id") or "").strip()
    agent_sha256 = str(entry.get("agent_sha256") or "").strip().lower()
    accepted_at = entry.get("accepted_at")
    if not hotkey or not submission_id or not agent_sha256 or not accepted_at:
        return None
    item: dict[str, Any] = {
        "hotkey": hotkey,
        "repo": "private-submission",
        "repo_full_name": f"private-submission/{submission_id}",
        "repo_url": None,
        "commit_sha": agent_sha256,
        "display_repo_full_name": "private-submission",
        "display_repo_url": None,
        "display_commit_sha": agent_sha256,
        "source": _PRIVATE_SOURCE,
        "accepted_at": str(accepted_at),
        "commitment": f"private-submission:{submission_id}:{agent_sha256}",
    }
    uid = entry.get("uid")
    if uid is not None:
        try:
            item["uid"] = int(uid)
        except (TypeError, ValueError):
            pass
    if entry.get("agent_username"):
        item["agent_username"] = entry["agent_username"]
    if entry.get("coldkey"):
        item["coldkey"] = entry["coldkey"]
    return item


def _ledger_uid_map(validate_root: Path) -> dict[str, int]:
    root = _ledger_root(validate_root)
    if not root.is_dir():
        return {}
    mapping: dict[str, int] = {}
    for entry in accepted_private_submission_entries(root=root):
        hotkey = str(entry.get("hotkey") or "")
        uid = entry.get("uid")
        if not hotkey or uid is None:
            continue
        try:
            mapping[hotkey] = int(uid)
        except (TypeError, ValueError):
            continue
    return mapping


def _fill_queue_uids(
    queue_items: list[dict[str, Any]],
    *,
    netuid: int | None = None,
    validate_root: Path | None = None,
) -> list[dict[str, Any]]:
    missing = [
        str(item.get("hotkey") or "")
        for item in queue_items
        if isinstance(item, dict) and item.get("hotkey") and item.get("uid") is None
    ]
    if not missing:
        return queue_items

    uid_map: dict[str, int] = {}
    if validate_root is not None:
        uid_map.update(_ledger_uid_map(validate_root))

    if hotkey_uid_map is not None:
        still_missing = [hotkey for hotkey in missing if hotkey not in uid_map]
        if still_missing:
            try:
                uid_map.update(hotkey_uid_map(netuid=netuid))
            except Exception:
                pass

    if not uid_map:
        return queue_items

    filled: list[dict[str, Any]] = []
    for item in queue_items:
        if not isinstance(item, dict):
            continue
        merged = dict(item)
        if merged.get("uid") is None:
            hotkey = str(merged.get("hotkey") or "")
            uid = uid_map.get(hotkey)
            if uid is not None:
                merged["uid"] = int(uid)
        filled.append(merged)
    return filled


def _participant_hotkeys(participant: Any) -> set[str]:
    if not isinstance(participant, dict):
        return set()
    hotkey = participant.get("hotkey")
    return {str(hotkey)} if hotkey else set()


def _excluded_hotkeys(*, status: dict[str, Any], state: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for key in ("seen_hotkeys", "retired_hotkeys", "disqualified_hotkeys"):
        values = state.get(key)
        if isinstance(values, list):
            excluded.update(str(hotkey) for hotkey in values if hotkey)

    locked = state.get("locked_commitments")
    if isinstance(locked, dict):
        excluded.update(str(hotkey) for hotkey in locked.keys())

    king = status.get("current_king")
    if not isinstance(king, dict):
        king = state.get("current_king")
    excluded.update(_participant_hotkeys(king))

    active = status.get("active_duel")
    if isinstance(active, dict):
        for field in ("king_hotkey", "challenger_hotkey", "hotkey"):
            if active.get(field):
                excluded.add(str(active[field]))
    else:
        raw_active = state.get("active_duel")
        if isinstance(raw_active, dict):
            for role in ("king", "challenger"):
                excluded.update(_participant_hotkeys(raw_active.get(role)))

    return excluded


def _latest_queue_accepted_at(queue_items: list[dict[str, Any]]) -> str | None:
    accepted_times = [
        str(item.get("accepted_at"))
        for item in queue_items
        if isinstance(item, dict) and item.get("accepted_at")
    ]
    return max(accepted_times) if accepted_times else None


def merge_acceptance_ledger_into_queue(
    *,
    status: dict[str, Any],
    validate_root: Path,
    state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return queue items from snapshot/state plus accepted ledger backlog."""
    if not isinstance(status, dict):
        return []

    queue_by_hotkey: dict[str, dict[str, Any]] = {}
    existing = status.get("queue")
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, dict) and item.get("hotkey"):
                queue_by_hotkey[str(item["hotkey"])] = dict(item)

    state_payload = state if isinstance(state, dict) else _read_json_dict(validate_root / "state.json")
    state_queue = state_payload.get("queue")
    if isinstance(state_queue, list):
        for item in state_queue:
            if isinstance(item, dict) and item.get("hotkey"):
                hotkey = str(item["hotkey"])
                queue_by_hotkey[hotkey] = {**queue_by_hotkey.get(hotkey, {}), **item}

    excluded = _excluded_hotkeys(status=status, state=state_payload)
    latest_accepted_at = _latest_queue_accepted_at(list(queue_by_hotkey.values()))
    root = _ledger_root(validate_root)
    if root.is_dir():
        for entry in accepted_private_submission_entries(root=root):
            hotkey = str(entry.get("hotkey") or "")
            accepted_at = str(entry.get("accepted_at") or "")
            if not hotkey or hotkey in queue_by_hotkey or hotkey in excluded:
                continue
            if latest_accepted_at and accepted_at <= latest_accepted_at:
                continue
            item = _queue_item_from_ledger_entry(entry)
            if item is not None:
                queue_by_hotkey[hotkey] = item

    merged = sorted(queue_by_hotkey.values(), key=_sort_key)
    netuid = status.get("netuid")
    try:
        netuid_value = int(netuid) if netuid is not None else 66
    except (TypeError, ValueError):
        netuid_value = 66
    return _fill_queue_uids(merged, netuid=netuid_value, validate_root=validate_root)


def augment_dashboard_status_queue(
    *,
    status: dict[str, Any],
    validate_root: Path,
) -> dict[str, Any]:
    if not isinstance(status, dict):
        return status
    merged_queue = merge_acceptance_ledger_into_queue(status=status, validate_root=validate_root)
    if merged_queue == status.get("queue"):
        return status
    return {**status, "queue": merged_queue}


def augment_dashboard_payload(
    payload: dict[str, Any],
    *,
    dashboard_data_path: str | Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    status = payload.get("status")
    if not isinstance(status, dict):
        return payload

    validate_root = Path(dashboard_data_path).resolve().parent
    new_status = augment_dashboard_status_queue(status=status, validate_root=validate_root)
    if new_status is status:
        return payload

    return {
        **payload,
        "updated_at": datetime.now(UTC).isoformat(),
        "status": new_status,
    }
