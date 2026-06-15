"""Cached hotkey -> uid lookups for dashboard and ledger backfill."""

from __future__ import annotations

import os
import time
from typing import Any

_CACHE: dict[str, Any] = {"map": {}, "expires_at": 0.0}


def hotkey_uid_map(*, netuid: int | None = None, ttl_seconds: int = 300) -> dict[str, int]:
    netuid_value = int(netuid or os.environ.get("VALIDATE_NETUID", 66))
    now = time.monotonic()
    cached = _CACHE.get("map")
    expires_at = float(_CACHE.get("expires_at") or 0)
    if ttl_seconds > 0 and isinstance(cached, dict) and now < expires_at:
        return dict(cached)

    import bittensor as bt

    network = (
        os.environ.get("VALIDATE_NETWORK")
        or os.environ.get("SUBTENSOR_NETWORK")
        or "finney"
    )
    subtensor = bt.SubtensorApi(network=network, websocket_shutdown_timer=0)
    neurons = subtensor.neurons.neurons_lite(netuid=netuid_value)
    mapping = {
        str(neuron.hotkey): int(neuron.uid)
        for neuron in neurons
        if getattr(neuron, "hotkey", None) is not None
    }
    _CACHE["map"] = mapping
    _CACHE["expires_at"] = now + max(ttl_seconds, 0)
    return dict(mapping)


def clear_hotkey_uid_cache() -> None:
    _CACHE["map"] = {}
    _CACHE["expires_at"] = 0.0
