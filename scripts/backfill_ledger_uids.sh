#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec /home/const/subnet66/.venv/bin/python - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/const/subnet66/tau/src")))
from private_submission import backfill_acceptance_ledger_uids

root = Path("/home/const/subnet66/tau/workspace/validate/netuid-66/private-submissions")
updated = backfill_acceptance_ledger_uids(root=root, netuid=66)
print(f"backfilled {updated} acceptance ledger uid(s)")
PY
