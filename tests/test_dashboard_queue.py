from __future__ import annotations

import json
from pathlib import Path

from dashboard_queue import augment_dashboard_payload, merge_acceptance_ledger_into_queue


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_merge_acceptance_ledger_adds_pending_submissions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard_queue.hotkey_uid_map",
        lambda **_: {
            "5PendingHotkey11111111111111111111111111111111": 42,
        },
    )
    validate_root = tmp_path / "netuid-66"
    ledger_root = validate_root / "private-submissions"
    ledger_root.mkdir(parents=True)

    _write_json(
        ledger_root / "_accepted_submissions.json",
        {
            "version": 1,
            "hotkeys": {
                "5ExistingHotkey1111111111111111111111111111111": {
                    "submission_id": "existing-sub",
                    "agent_sha256": "aa" * 32,
                    "registration_block": 100,
                    "accepted_at": "2026-06-15T09:00:00+00:00",
                },
                "5OldSeenHotkey111111111111111111111111111111111": {
                    "submission_id": "old-sub",
                    "agent_sha256": "cc" * 32,
                    "registration_block": 99,
                    "accepted_at": "2026-06-15T08:00:00+00:00",
                },
                "5PendingHotkey11111111111111111111111111111111": {
                    "submission_id": "pending-sub",
                    "agent_sha256": "bb" * 32,
                    "registration_block": 101,
                    "accepted_at": "2026-06-15T13:40:00+00:00",
                    "agent_username": "pending-miner",
                    "uid": 42,
                },
            },
        },
    )

    status = {
        "queue": [
            {
                "uid": 7,
                "hotkey": "5ExistingHotkey1111111111111111111111111111111",
                "repo": "private-submission",
                "accepted_at": "2026-06-15T09:00:00+00:00",
            }
        ],
        "current_king": {"uid": 1, "hotkey": "5KingHotkey111111111111111111111111111111111"},
        "active_duel": {"challenger_hotkey": "5ActiveHotkey1111111111111111111111111111111"},
    }
    _write_json(
        validate_root / "state.json",
        {
            "seen_hotkeys": ["5SeenHotkey111111111111111111111111111111111"],
            "queue": status["queue"],
            "current_king": status["current_king"],
            "active_duel": {
                "challenger": {"hotkey": "5ActiveHotkey1111111111111111111111111111111"},
            },
        },
    )

    merged = merge_acceptance_ledger_into_queue(status=status, validate_root=validate_root)
    hotkeys = [item["hotkey"] for item in merged]

    assert hotkeys == [
        "5ExistingHotkey1111111111111111111111111111111",
        "5PendingHotkey11111111111111111111111111111111",
    ]
    assert "5OldSeenHotkey111111111111111111111111111111111" not in hotkeys
    pending = merged[1]
    assert pending["uid"] == 42
    assert pending["agent_username"] == "pending-miner"
    assert pending["repo_full_name"] == "private-submission/pending-sub"


def test_augment_dashboard_payload_updates_timestamp(tmp_path: Path) -> None:
    validate_root = tmp_path / "netuid-66"
    ledger_root = validate_root / "private-submissions"
    ledger_root.mkdir(parents=True, exist_ok=True)
    dashboard_path = validate_root / "dashboard_data.json"

    _write_json(
        ledger_root / "_accepted_submissions.json",
        {
            "version": 1,
            "hotkeys": {
                "5ExistingHotkey1111111111111111111111111111111": {
                    "submission_id": "existing-sub",
                    "agent_sha256": "aa" * 32,
                    "registration_block": 100,
                    "accepted_at": "2026-06-15T09:00:00+00:00",
                },
                "5PendingHotkey11111111111111111111111111111111": {
                    "submission_id": "pending-sub",
                    "agent_sha256": "bb" * 32,
                    "registration_block": 101,
                    "accepted_at": "2026-06-15T13:40:00+00:00",
                },
            },
        },
    )

    payload = {
        "updated_at": "2026-06-15T13:32:18+00:00",
        "status": {
            "queue": [
                {
                    "hotkey": "5ExistingHotkey1111111111111111111111111111111",
                    "accepted_at": "2026-06-15T09:00:00+00:00",
                }
            ]
        },
    }

    augmented = augment_dashboard_payload(payload, dashboard_data_path=dashboard_path)

    assert augmented["updated_at"] != payload["updated_at"]
    assert len(augmented["status"]["queue"]) == 2
