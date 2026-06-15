from __future__ import annotations

import json
from pathlib import Path

from dashboard_queue import _fill_queue_uids


def test_fill_queue_uids_uses_chain_map(monkeypatch, tmp_path: Path) -> None:
    validate_root = tmp_path / "netuid-66"
    ledger_root = validate_root / "private-submissions"
    ledger_root.mkdir(parents=True)
    (ledger_root / "_accepted_submissions.json").write_text(
        json.dumps(
            {
                "version": 1,
                "hotkeys": {
                    "5Hotkey1111111111111111111111111111111111111": {
                        "submission_id": "sub-1",
                        "agent_sha256": "aa" * 32,
                        "registration_block": 100,
                        "accepted_at": "2026-06-15T13:40:00+00:00",
                        "uid": 77,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dashboard_queue.hotkey_uid_map",
        lambda **_: {},
    )
    queue = [
        {"hotkey": "5Hotkey1111111111111111111111111111111111111", "uid": None},
        {"hotkey": "5Other11111111111111111111111111111111111111", "uid": 3},
    ]
    filled = _fill_queue_uids(queue, netuid=66, validate_root=validate_root)
    assert filled[0]["uid"] == 77
    assert filled[1]["uid"] == 3
