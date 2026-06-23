#!/usr/bin/env python3
"""One-shot set_weights using production validator state and distribution."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RunConfig
from validate import (  # noqa: E402
    ValidatorState,
    _effective_recent_kings,
    _incumbent_allowed_by_mode,
    _king_emission_shares,
    _maybe_set_weights,
    _resolve_weight_uid,
    _BURN_KING_UID,
)
from tau import bittensor as bt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("test_set_weights")


def _build_config(workspace_root: Path) -> RunConfig:
    return RunConfig(
        workspace_root=workspace_root,
        validate_netuid=66,
        validate_wallet_name="sn66_owner",
        validate_wallet_hotkey="default",
        validate_wallet_path=str(Path.home() / ".bittensor" / "wallets"),
        validate_king_window_size=5,
        validate_weight_interval_blocks=360,
    )


def _preview(subtensor, config: RunConfig, state: ValidatorState) -> dict:
    shares = _king_emission_shares(config.validate_king_window_size)
    neurons = list(subtensor.neurons.neurons_lite(config.validate_netuid))
    uids = [int(n.uid) for n in neurons]
    uid_by_hotkey = {
        str(n.hotkey): int(n.uid) for n in neurons if getattr(n, "hotkey", None) is not None
    }
    weights_by_uid: dict[int, float] = {u: 0.0 for u in uids}
    burn_share = 0.0
    slots: list[dict] = []
    recent = _effective_recent_kings(state)
    for i, share in enumerate(shares):
        sub = recent[i] if i < len(recent) else None
        uid = None
        label = "burn"
        if sub is not None and _incumbent_allowed_by_mode(config, sub):
            uid = _resolve_weight_uid(submission=sub, uid_by_hotkey=uid_by_hotkey)
        if uid is not None and sub is not None:
            weights_by_uid[uid] += share
            label = sub.hotkey
        else:
            burn_share += share
            uid = _BURN_KING_UID
            weights_by_uid[_BURN_KING_UID] += share
        slots.append(
            {
                "slot": i,
                "share": share,
                "uid": uid,
                "hotkey": None if label == "burn" else label,
                "destination": "burn" if label == "burn" else "king",
            }
        )
    nonzero = {str(u): round(weights_by_uid[u], 6) for u in uids if weights_by_uid[u] > 0}
    return {
        "shares": shares,
        "burn_share": burn_share,
        "slots": slots,
        "nonzero_weights": nonzero,
        "uid_order_len": len(uids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/home/const/subnet66/tau"),
        help="Tau workspace root",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Validator state.json (default: workspace/validate/netuid-66/state.json)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit weights on-chain (default: preview only)",
    )
    args = parser.parse_args()

    workspace = args.workspace_root.expanduser().resolve()
    state_path = args.state_path or (workspace / "workspace" / "validate" / "netuid-66" / "state.json")
    payload = json.loads(state_path.read_text())
    state = ValidatorState.from_dict(payload)
    config = _build_config(workspace)

    subtensor = bt.SubtensorApi(websocket_shutdown_timer=0)
    current_block = int(subtensor.block)
    preview = _preview(subtensor, config, state)

    print("=== weight preview ===")
    print(json.dumps(preview, indent=2))
    print(f"current_block={current_block}")
    print(f"last_weight_block={state.last_weight_block}")

    if not args.submit:
        print("Dry run only. Re-run with --submit to set weights on-chain.")
        return 0

    before = state.last_weight_block
    _maybe_set_weights(
        subtensor=subtensor,
        config=config,
        state=state,
        current_block=current_block,
        force=True,
    )
    print(f"last_weight_block: {before} -> {state.last_weight_block}")
    return 0 if state.last_weight_block == current_block else 1


if __name__ == "__main__":
    raise SystemExit(main())
