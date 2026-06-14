#!/home/const/subnet66/.venv/bin/python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import RunConfig  # noqa: E402
from validate import (  # noqa: E402
    ValidatorState,
    ValidatorSubmission,
    _build_github_merge_client,
    _is_private_submission,
    _load_state,
    _prepare_validate_paths,
    _publish_promoted_private_submission,
    _record_king_transition,
    _retire_hotkey,
    _save_state,
)


def _submission_from_duel(duels_dir: Path, duel_id: int) -> ValidatorSubmission:
    duel_path = duels_dir / f"{duel_id:06d}.json"
    payload = json.loads(duel_path.read_text(encoding="utf-8"))
    challenger = payload.get("challenger")
    if not isinstance(challenger, dict):
        raise RuntimeError(f"duel {duel_id} has no challenger payload")
    return ValidatorSubmission.from_dict(challenger)


def _patch_duel_record(duels_dir: Path, duel_id: int, *, king_after: ValidatorSubmission) -> None:
    duel_path = duels_dir / f"{duel_id:06d}.json"
    payload = json.loads(duel_path.read_text(encoding="utf-8"))
    payload["king_replaced"] = True
    payload["king_after"] = king_after.to_dict()
    payload["confirmation_failure_reason"] = None
    duel_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manually publish a duel winner and install it as king.")
    parser.add_argument("--duel-id", type=int, required=True, help="Primary duel id whose challenger becomes king")
    parser.add_argument("--confirmation-duel-id", type=int, default=None, help="Optional confirmation duel id to patch")
    parser.add_argument("--netuid", type=int, default=66)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--king-window", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = RunConfig(
        workspace_root=args.workspace_root.resolve(),
        validate_netuid=args.netuid,
        validate_publish_repo="unarbos/ninja",
        validate_publish_base="main",
    )
    paths = _prepare_validate_paths(config.validate_root)
    state = _load_state(paths.state_path)
    candidate = _submission_from_duel(paths.duels_dir, args.duel_id)
    if not _is_private_submission(candidate):
        raise SystemExit(f"duel {args.duel_id} challenger is not a private submission")

    print(f"Candidate uid={candidate.uid} hotkey={candidate.hotkey[:16]}...")
    print(f"Commitment: {candidate.commitment}")

    github_client = _build_github_merge_client(config)
    try:
        published = _publish_promoted_private_submission(
            github_client=github_client,
            config=config,
            submission=candidate,
        )
    finally:
        github_client.close()

    if _is_private_submission(published):
        raise SystemExit("GitHub publication failed; candidate is still private")

    print(f"Published to {published.repo_full_name}@{published.commit_sha[:12]}")

    if args.dry_run:
        print("Dry run: state not modified")
        return 0

    old_king = state.current_king
    if old_king is not None and old_king.hotkey != published.hotkey:
        _retire_hotkey(state, old_king.hotkey)
    _record_king_transition(state, published, window=args.king_window)
    state.active_duel = None
    state.locked_commitments[published.hotkey] = published.commitment
    state.commitment_blocks_by_hotkey[published.hotkey] = int(published.commitment_block or 0)
    _save_state(paths.state_path, state)

    _patch_duel_record(paths.duels_dir, args.duel_id, king_after=published)
    confirmation_id = args.confirmation_duel_id
    if confirmation_id is None:
        duel_payload = json.loads((paths.duels_dir / f"{args.duel_id:06d}.json").read_text(encoding="utf-8"))
        raw_confirmation = duel_payload.get("confirmation_duel_id")
        confirmation_id = int(raw_confirmation) if raw_confirmation else None
    if confirmation_id is not None:
        _patch_duel_record(paths.duels_dir, confirmation_id, king_after=published)

    print(
        f"Installed king uid={published.uid} on {published.repo_full_name}@{published.commit_sha[:12]}; "
        f"cleared active duel; patched duel(s) {args.duel_id}"
        + (f", {confirmation_id}" if confirmation_id is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
