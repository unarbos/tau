"""Tests for sync_pm2_dump.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_pm2_dump.sh"


def test_sync_script_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_sync_script_does_not_auto_start() -> None:
    text = SCRIPT.read_text()
    assert "pm2 start" not in text
    assert "pm2 restart" not in text
    assert "pm2 save" in text
