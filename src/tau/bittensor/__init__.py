"""Mockable bittensor wrapper.

Default (live) mode delegates to the real bittensor package.
Call `init()` before use to switch modes:

    from tau import bittensor as bt
    bt.init(mode="debug")           # log chain ops to console / file
    bt.init(mode="test")            # silent in-memory mocks; no chain
    bt.init(mode="debug", debug_output_path="/tmp/bt.jsonl")

In test/debug mode the mock chain is seeded from an optional snapshot (a dict or
a path to a JSON file) so the validator can run fully offline:

    bt.init(mode="test", snapshot="/tmp/chain.json")

Snapshot schema::

    {
      "block": 8200000,
      "validator": {"hotkey": "5...", "uid": 1},
      "miners": [
        {"hotkey": "5...", "coldkey": "5...", "uid": 2,
         "registration_block": 7950000,
         "commitment": "unarbos/ninja@<sha>"}   # optional (omit for private-only)
      ]
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

log = logging.getLogger(__name__)

_BURN_UID = 0

_mode: Literal["live", "test", "debug"] = "live"
_debug_output_path: Path | None = None
_snapshot: ChainSnapshot | None = None


def init(
    mode: Literal["live", "test", "debug"] = "live",
    debug_output_path: str | Path | None = None,
    snapshot: dict[str, Any] | str | Path | None = None,
) -> None:
    """Switch the bittensor module mode.

    Args:
        mode: "live" uses the real chain; "debug" logs chain ops to console or
            *debug_output_path*; "test" silently succeeds with in-memory mocks.
        debug_output_path: File to append JSON-lines debug output (debug mode only).
            Defaults to console via the module logger.
        snapshot: In test/debug mode, the in-memory chain state (dict or JSON
            path). When omitted, an empty chain (block 1, burn uid 0 only) is used.
    """
    global _mode, _debug_output_path, _snapshot
    _mode = mode
    _debug_output_path = Path(debug_output_path) if debug_output_path else None
    _snapshot = ChainSnapshot.load(snapshot) if mode in ("test", "debug") else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _debug_log(data: dict[str, Any]) -> None:
    if _debug_output_path:
        # JSON-lines: one compact object per line so the file is replayable.
        with open(_debug_output_path, "a") as fh:
            fh.write(json.dumps(data, default=str) + "\n")
    else:
        log.info("[bittensor-debug]\n%s", json.dumps(data, indent=2, default=str))


def _active_snapshot() -> ChainSnapshot:
    return _snapshot if _snapshot is not None else ChainSnapshot.empty()


# ---------------------------------------------------------------------------
# In-memory chain snapshot
# ---------------------------------------------------------------------------

class ChainSnapshot:
    """Read-only view of a seeded mock chain; safe to share across threads."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.block: int = int(data.get("block", 1))
        validator = data.get("validator") or {}
        self.validator_hotkey: str | None = validator.get("hotkey")
        self.validator_uid: int | None = (
            int(validator["uid"]) if validator.get("uid") is not None else None
        )
        self.miners: list[dict[str, Any]] = list(data.get("miners") or [])

        self._uid_by_hotkey: dict[str, int] = {}
        self._coldkey_by_hotkey: dict[str, str] = {}
        self._regblock_by_uid: dict[int, int] = {}
        self._commitment_by_hotkey: dict[str, str] = {}
        self._commitment_block_by_hotkey: dict[str, int] = {}

        if self.validator_hotkey is not None and self.validator_uid is not None:
            self._uid_by_hotkey[self.validator_hotkey] = self.validator_uid

        for miner in self.miners:
            hotkey = str(miner["hotkey"])
            uid = int(miner["uid"])
            self._uid_by_hotkey[hotkey] = uid
            if miner.get("coldkey") is not None:
                self._coldkey_by_hotkey[hotkey] = str(miner["coldkey"])
            if miner.get("registration_block") is not None:
                self._regblock_by_uid[uid] = int(miner["registration_block"])
            commitment = miner.get("commitment")
            if commitment is not None:
                self._commitment_by_hotkey[hotkey] = str(commitment)
                self._commitment_block_by_hotkey[hotkey] = int(
                    miner.get("commitment_block", miner.get("registration_block", self.block))
                )

    @classmethod
    def load(cls, source: dict[str, Any] | str | Path | None) -> ChainSnapshot:
        if source is None:
            return cls.empty()
        if isinstance(source, dict):
            return cls(source)
        return cls(json.loads(Path(source).read_text()))

    @classmethod
    def empty(cls) -> ChainSnapshot:
        return cls({"block": 1, "miners": []})

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        return self._uid_by_hotkey.get(str(hotkey))

    def coldkey_for_hotkey(self, hotkey: str) -> str | None:
        return self._coldkey_by_hotkey.get(str(hotkey))

    def registration_block_for_uid(self, uid: int) -> int | None:
        return self._regblock_by_uid.get(int(uid))

    def commitments(self) -> dict[str, str]:
        return dict(self._commitment_by_hotkey)

    def commitment_block_for_hotkey(self, hotkey: str) -> int | None:
        return self._commitment_block_by_hotkey.get(str(hotkey))

    def neuron_uids(self) -> list[int]:
        uids = {_BURN_UID}
        uids.update(self._uid_by_hotkey.values())
        return sorted(uids)

    def hotkey_for_uid(self, uid: int) -> str | None:
        for hotkey, value in self._uid_by_hotkey.items():
            if value == uid:
                return hotkey
        return None


# ---------------------------------------------------------------------------
# Mock types
# ---------------------------------------------------------------------------

class _MockWeightsResult:
    success = True
    message = "bittensor mock: weights not submitted to chain"


class _MockExtrinsics:
    def set_weights(
        self,
        wallet: Any,
        netuid: int,
        uids: Any,
        weights: Any,
        **kwargs: Any,
    ) -> _MockWeightsResult:
        if _mode == "debug":
            _debug_log({
                "action": "set_weights",
                "netuid": netuid,
                "wallet_name": getattr(wallet, "name", None),
                "wallet_hotkey": getattr(wallet, "hotkey_str", None),
                "uids": list(uids),
                "weights": [float(w) for w in weights],
            })
        return _MockWeightsResult()


class _MockCommitments:
    def get_all_revealed_commitments(self, netuid: int) -> dict[str, Any]:
        # The validator normalizes each hotkey's entries to (block, commitment).
        snap = _active_snapshot()
        return {
            hotkey: [(snap.commitment_block_for_hotkey(hotkey) or snap.block, commitment)]
            for hotkey, commitment in snap.commitments().items()
        }

    def get_all_commitments(self, netuid: int) -> dict[str, str]:
        return _active_snapshot().commitments()

    def get_commitment_metadata(self, netuid: int, hotkey: str) -> list[dict[str, int]]:
        snap = _active_snapshot()
        block = snap.commitment_block_for_hotkey(hotkey)
        return [{"block": block}] if block is not None else []


class _MockSubnets:
    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int, **kwargs: Any) -> int | None:
        return _active_snapshot().uid_for_hotkey(hotkey)


class _MockNeurons:
    def neurons_lite(self, netuid: int, **kwargs: Any) -> list[SimpleNamespace]:
        snap = _active_snapshot()
        return [
            SimpleNamespace(
                uid=uid,
                hotkey=snap.hotkey_for_uid(uid),
                coldkey=snap.coldkey_for_hotkey(snap.hotkey_for_uid(uid) or ""),
            )
            for uid in snap.neuron_uids()
        ]


class _MockSubstrate:
    def query(
        self,
        module: str,
        storage_function: str,
        params: list[Any] | None = None,
        block_hash: Any = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        snap = _active_snapshot()
        params = params or []
        if storage_function == "BlockAtRegistration":
            uid = int(params[1]) if len(params) > 1 else None
            value = snap.registration_block_for_uid(uid) if uid is not None else None
            return SimpleNamespace(value=value)
        if storage_function == "Owner":
            hotkey = str(params[0]) if params else ""
            return SimpleNamespace(value=snap.coldkey_for_hotkey(hotkey))
        return SimpleNamespace(value=None)


class _MockSubtensor:
    def __init__(self, network: str | None = None, **kwargs: Any) -> None:
        self.extrinsics = _MockExtrinsics()
        self.commitments = _MockCommitments()
        self.subnets = _MockSubnets()
        self.neurons = _MockNeurons()
        self.substrate = _MockSubstrate()
        if _mode == "debug":
            _debug_log({"action": "SubtensorApi_open", "network": network})

    # The validator uses the subtensor as a context manager.
    def __enter__(self) -> _MockSubtensor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    @property
    def block(self) -> int:
        return _active_snapshot().block

    def determine_block_hash(self, block: int | None = None) -> None:
        return None


class _MockWallet:
    def __init__(self, name: str | None = None, hotkey: str | None = None, path: str | None = None) -> None:
        self.name = name
        self.hotkey_str = hotkey
        self.path = path


class _MockKeypair:
    def __init__(self, ss58_address: str) -> None:
        self.ss58_address = ss58_address

    def verify(self, message: Any, signature: Any) -> bool:
        return True


# ---------------------------------------------------------------------------
# Public API — mirrors real bittensor surface used by validate.py
# ---------------------------------------------------------------------------

def Wallet(name: str | None = None, hotkey: str | None = None, path: str | None = None) -> Any:
    if _mode == "live":
        import bittensor as _bt
        return _bt.Wallet(name=name, hotkey=hotkey, path=path)
    return _MockWallet(name=name, hotkey=hotkey, path=path)


def Keypair(ss58_address: str) -> Any:
    if _mode == "live":
        import bittensor as _bt
        return _bt.Keypair(ss58_address=ss58_address)
    return _MockKeypair(ss58_address=ss58_address)


def SubtensorApi(network: str | None = None, **kwargs: Any) -> Any:
    if _mode == "live":
        import bittensor as _bt
        if network is not None:
            return _bt.SubtensorApi(network=network, **kwargs)
        return _bt.SubtensorApi(**kwargs)
    return _MockSubtensor(network=network, **kwargs)
