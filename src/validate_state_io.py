from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2 import duel_to_summary, publish_duel_data, publish_duel_index
from workspace import write_json

log = logging.getLogger("swe-eval.validate")


@dataclass(slots=True)
class ManualRetestSeed:
    good_rounds: list[Any]
    target_round_count: int
    error_round_count: int
    prior_task_names: set[str]


def _load_state(path: Path, *, state_cls: type[Any]) -> Any:
    if not path.exists():
        return state_cls()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid state file: {path}")
    return state_cls.from_dict(payload)


def _reconcile_state_with_duel_history(state: Any, duels_dir: Path) -> bool:
    """Recover monotonic state from durable duel result files."""
    max_duel_id = 0
    completed_hotkeys: set[str] = set()
    completed_commitments: dict[str, str] = {}
    completed_blocks: dict[str, int] = {}

    for duel_path in duels_dir.glob("*.json"):
        try:
            payload = json.loads(duel_path.read_text())
        except Exception:
            log.exception("Failed to load duel history file %s during state recovery", duel_path)
            continue
        if not isinstance(payload, dict):
            continue

        try:
            duel_id = int(payload.get("duel_id", duel_path.stem))
        except (TypeError, ValueError):
            try:
                duel_id = int(duel_path.stem)
            except ValueError:
                duel_id = 0
        max_duel_id = max(max_duel_id, duel_id)

        challenger = payload.get("challenger")
        if not isinstance(challenger, dict):
            continue
        hotkey = str(challenger.get("hotkey") or "")
        if not hotkey:
            continue
        completed_hotkeys.add(hotkey)

        commitment = challenger.get("commitment")
        if commitment:
            completed_commitments.setdefault(hotkey, str(commitment))
        try:
            completed_blocks.setdefault(hotkey, int(challenger.get("commitment_block")))
        except (TypeError, ValueError):
            pass

    changed = False
    if max_duel_id >= state.next_duel_index:
        state.next_duel_index = max_duel_id + 1
        changed = True

    removed_from_queue = 0
    if completed_hotkeys:
        before = len(state.queue)
        state.queue = [
            s
            for s in state.queue
            if s.hotkey not in completed_hotkeys or s.manual_retest_of_duel_id is not None
        ]
        removed_from_queue = before - len(state.queue)
        changed = changed or removed_from_queue > 0

        for hotkey in sorted(completed_hotkeys):
            if hotkey not in state.seen_hotkeys:
                state.seen_hotkeys.append(hotkey)
                changed = True
        for hotkey, commitment in completed_commitments.items():
            if hotkey not in state.locked_commitments:
                state.locked_commitments[hotkey] = commitment
                changed = True
        for hotkey, block in completed_blocks.items():
            if hotkey not in state.commitment_blocks_by_hotkey:
                state.commitment_blocks_by_hotkey[hotkey] = block
                changed = True

    if changed:
        log.info(
            "Reconciled validator state with duel history: next_duel_index=%d, "
            "completed_hotkeys=%d, removed_queue_entries=%d",
            state.next_duel_index,
            len(completed_hotkeys),
            removed_from_queue,
        )
    return changed


def _save_state(path: Path, state: Any) -> None:
    write_json(path, state.to_dict())


def _write_duel(paths: Any, duel: Any) -> None:
    write_json(paths.duels_dir / f"{duel.duel_id:06d}.json", duel.to_dict())


def _manual_retest_seed_from_history(
    duels_dir: Path,
    duel_id: int | None,
    *,
    validation_round_result_cls: type[Any],
) -> ManualRetestSeed | None:
    if duel_id is None:
        return None
    path = duels_dir / f"{duel_id:06d}.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        log.warning("Referenced duel %d history file is missing at %s", duel_id, path)
        return None
    except Exception:
        log.exception("Failed to load referenced duel %d rounds from %s", duel_id, path)
        return None
    if not isinstance(payload, dict):
        return None

    prior_task_names: set[str] = set()
    task_names = payload.get("task_names")
    if isinstance(task_names, list):
        prior_task_names.update(str(name) for name in task_names if name)

    raw_rounds = payload.get("rounds")
    if not isinstance(raw_rounds, list):
        return None

    good_rounds: list[Any] = []
    parsed_round_count = 0
    error_round_count = 0
    for item in raw_rounds:
        if not isinstance(item, dict):
            continue
        task_name = item.get("task_name")
        if task_name:
            prior_task_names.add(str(task_name))
        try:
            round_result = validation_round_result_cls(**item)
        except TypeError:
            continue
        parsed_round_count += 1
        if round_result.error is None:
            good_rounds.append(round_result)
        else:
            error_round_count += 1

    if parsed_round_count <= 0:
        return None
    return ManualRetestSeed(
        good_rounds=good_rounds,
        target_round_count=parsed_round_count,
        error_round_count=error_round_count,
        prior_task_names=prior_task_names,
    )


def _duel_task_names_from_history(duels_dir: Path, duel_id: int | None) -> set[str]:
    if duel_id is None:
        return set()
    path = duels_dir / f"{duel_id:06d}.json"
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        log.warning("Referenced duel %d history file is missing at %s", duel_id, path)
        return set()
    except Exception:
        log.exception("Failed to load referenced duel %d task names from %s", duel_id, path)
        return set()
    if not isinstance(payload, dict):
        return set()

    names: set[str] = set()
    task_names = payload.get("task_names")
    if isinstance(task_names, list):
        names.update(str(name) for name in task_names if name)
    rounds = payload.get("rounds")
    if isinstance(rounds, list):
        for round_payload in rounds:
            if not isinstance(round_payload, dict):
                continue
            task_name = round_payload.get("task_name")
            if task_name:
                names.add(str(task_name))
    return names


def _load_dashboard_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        log.exception("Failed to load dashboard history; starting fresh")
        return []


def _reconcile_dashboard_history_with_duels(
    history: list[dict[str, Any]],
    duels_dir: Path,
    *,
    duel_to_summary_fn=duel_to_summary,
) -> bool:
    by_duel_id: dict[int, dict[str, Any]] = {}
    unknown_id_entries: list[dict[str, Any]] = []
    changed = False

    for entry in history:
        if not isinstance(entry, dict):
            changed = True
            continue
        try:
            duel_id = int(entry["duel_id"])
        except (KeyError, TypeError, ValueError):
            unknown_id_entries.append(entry)
            continue
        if duel_id in by_duel_id:
            changed = True
            continue
        by_duel_id[duel_id] = entry

    added = 0
    for duel_path in duels_dir.glob("*.json"):
        try:
            duel_dict = json.loads(duel_path.read_text())
        except Exception:
            log.exception("Failed to load duel history file %s during dashboard recovery", duel_path)
            continue
        if not isinstance(duel_dict, dict):
            continue
        try:
            duel_id = int(duel_dict.get("duel_id", duel_path.stem))
        except (TypeError, ValueError):
            try:
                duel_id = int(duel_path.stem)
            except ValueError:
                continue
        if duel_id in by_duel_id:
            continue
        by_duel_id[duel_id] = duel_to_summary_fn(duel_dict)
        added += 1
        changed = True

    if not changed:
        return False

    history[:] = unknown_id_entries + [by_duel_id[duel_id] for duel_id in sorted(by_duel_id)]
    log.info(
        "Reconciled dashboard history with duel files: entries=%d, added=%d",
        len(history),
        added,
    )
    return True


def _upsert_dashboard_history_summary(history: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    try:
        duel_id = int(summary["duel_id"])
    except (KeyError, TypeError, ValueError):
        history.append(summary)
        return True

    for index, entry in enumerate(history):
        if not isinstance(entry, dict):
            continue
        try:
            entry_duel_id = int(entry["duel_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if entry_duel_id == duel_id:
            history[index] = summary
            return False

    history.append(summary)
    return True


def _replay_local_duel_files_to_r2(
    paths: Any,
    dashboard_history: list[dict[str, Any]],
    *,
    publish_duel_data_fn=publish_duel_data,
    publish_duel_index_fn=publish_duel_index,
) -> None:
    duel_paths = sorted(paths.duels_dir.glob("*.json"), reverse=True)
    if not duel_paths:
        return

    published = 0
    failed = 0
    consecutive_failures = 0
    latest_duel_dict: dict[str, Any] | None = None
    for duel_path in duel_paths:
        try:
            duel_dict = json.loads(duel_path.read_text())
        except Exception:
            log.exception("R2 replay: failed to load local duel file %s", duel_path)
            continue
        if not isinstance(duel_dict, dict):
            continue
        try:
            duel_id = int(duel_dict.get("duel_id", duel_path.stem))
        except (TypeError, ValueError):
            try:
                duel_id = int(duel_path.stem)
            except ValueError:
                continue
        if latest_duel_dict is None:
            latest_duel_dict = duel_dict
        try:
            ok = publish_duel_data_fn(duel_id=duel_id, duel_dict=duel_dict)
        except Exception:
            log.exception("R2 replay: failed to publish local duel file %s", duel_path)
            ok = False
        if ok:
            published += 1
            consecutive_failures = 0
        else:
            failed += 1
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log.warning(
                    "R2 replay: stopping after %d consecutive duel publish failure(s)",
                    consecutive_failures,
                )
                break

    try:
        index_ok = publish_duel_index_fn(
            duel_history=dashboard_history,
            latest_duel_dict=latest_duel_dict,
        )
    except Exception:
        log.exception("R2 replay: failed to publish duel index")
        index_ok = False
    log.info(
        "R2 replay complete: published=%d failed=%d index=%s",
        published,
        failed,
        index_ok,
    )


def _save_dashboard_history(path: Path, history: list[Any]) -> None:
    write_json(path, history)
