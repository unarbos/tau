#!/usr/bin/env python3
"""Boot the real validate_loop_run with no network, for one duel.

The chain (bittensor), R2, the TaoMarketCap API, and HuggingFace are faked; the
king and challenger still run as real Docker solves judged by a real OpenRouter
key. Everything is synthesized from local inputs (a ninja repo clone, a
challenger agent.py, and a task archive), so no production validator state is
needed. The mocks come from src/tau:

  - chain:  tau.bittensor.init(mode="test", snapshot=...)
  - github: tau.io.github.LocalGitHubClient over a local clone (file:// repo_url)
  - R2:     tau.io.r2.LocalS3Client under <workspace>/r2; HF and TMC stay off
  - tasks:  expanded from the HF tasks/ archive into a pre-filled pool

--tasks must point at the tasks/ archive (full task definitions), not the
rollouts/ dataset: rollouts only record king/challenger solutions and lack the
repo trees a fresh solve needs. Re-judge those with src/replay_duels.py.

    OPENROUTER_API_KEY=... python scripts/dry_run_validator.py \
        --ninja-repo PATH --tasks tasks/primary/<hour>_primary.jsonl

The duel needs Docker + an OpenRouter key; the seeding helpers are covered
without Docker by tests/test_dry_run_validator.py.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tau.bittensor as bt  # noqa: E402
from config import RunConfig  # noqa: E402
from private_submission import (  # noqa: E402
    PrivateSubmissionCheckResult,
    SubmissionCheck,
    record_private_submission_acceptance,
    write_private_submission_bundle,
)

log = logging.getLogger("dry-run-validator")

_BURN_KING_HOTKEY = "burn-uid-0"
_BURN_KING_UID = 0
_BURN_KING_COMMITMENT = "burn:uid-0"
_MINER_REPO = "unarbos/ninja"

# A minimal challenger agent used when --challenger-agent is omitted. It must
# satisfy the private-submission scope guard (define solve(...), stdlib only).
_DEFAULT_AGENT_PY = '''\
"""Dry-run challenger agent."""


def solve(repo_path, issue, model, api_base, api_key, **kwargs):
    return ""
'''


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Chain snapshot synthesis
# ---------------------------------------------------------------------------

def synthesize_chain_snapshot(
    *,
    validator_hotkey: str,
    validator_uid: int,
    challenger_hotkey: str,
    challenger_coldkey: str,
    challenger_uid: int,
    registration_block: int,
    block: int,
) -> dict[str, Any]:
    """Mock-chain state covering the validator and challenger hotkeys.

    Registers their uids, the challenger's coldkey owner, and registration
    blocks so uid lookups, the substrate queries, and identity checks pass.
    """
    return {
        "block": block,
        "validator": {"hotkey": validator_hotkey, "uid": validator_uid},
        "miners": [
            {
                "hotkey": challenger_hotkey,
                "coldkey": challenger_coldkey,
                "uid": challenger_uid,
                "registration_block": registration_block,
            },
        ],
    }


# ---------------------------------------------------------------------------
# State + king seeding
# ---------------------------------------------------------------------------

def seed_burn_king_state(*, validate_root: Path, ninja_repo: Path, sha: str) -> Path:
    """Write state.json with a burn king served from the local ninja clone.

    The validator never builds a king itself (_ensure_king is a no-op), so it
    must be pre-seeded or no duel ever starts.
    """
    validate_root.mkdir(parents=True, exist_ok=True)
    king = {
        "hotkey": _BURN_KING_HOTKEY,
        "uid": _BURN_KING_UID,
        "repo_full_name": _MINER_REPO,
        "repo_url": f"file://{ninja_repo}",
        "commit_sha": sha,
        "commitment": _BURN_KING_COMMITMENT,
        "commitment_block": 1,
        "source": "burn",
    }
    state = {
        "current_king": king,
        "queue": [],
        "next_task_index": 1,
        "next_duel_index": 1,
    }
    state_path = validate_root / "state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state_path


# ---------------------------------------------------------------------------
# Private submission (challenger) seeding
# ---------------------------------------------------------------------------

def seed_private_submission(
    *,
    root: Path,
    agent_py_text: str,
    hotkey: str,
    coldkey: str,
    registration_block: int,
    submission_id: str | None = None,
    agent_username: str = "dry-run-challenger",
) -> tuple[str, str]:
    """Write an accepted challenger bundle (agent.py + check_result + ledger).

    Returns (submission_id, agent_sha256). The signature is a placeholder; the
    mock bt.Keypair.verify accepts any non-empty value.
    """
    root.mkdir(parents=True, exist_ok=True)
    agent_sha = hashlib.sha256(agent_py_text.encode("utf-8")).hexdigest()
    if submission_id is None:
        submission_id = f"{hotkey[:16]}-{agent_sha[:16]}"
    check_result = PrivateSubmissionCheckResult(
        accepted=True,
        agent_sha256=agent_sha,
        checks={
            name: SubmissionCheck(name=name, status="passed", summary="dry-run seed")
            for name in ("agent_smoke", "scope_guard", "openrouter_judge")
        },
    )
    write_private_submission_bundle(
        root=root,
        submission_id=submission_id,
        hotkey=hotkey,
        agent_py=agent_py_text,
        check_result=check_result,
        signature="dry-run-signature",
        registration_block=registration_block,
        agent_username=agent_username,
        coldkey=coldkey,
        coldkey_signature="dry-run-coldkey-signature",
        overwrite=True,
    )
    record_private_submission_acceptance(
        root=root,
        hotkey=hotkey,
        submission_id=submission_id,
        agent_sha256=agent_sha,
        registration_block=registration_block,
        agent_username=agent_username,
        coldkey=coldkey,
        coldkey_signature="dry-run-coldkey-signature",
    )
    return submission_id, agent_sha


# ---------------------------------------------------------------------------
# Saved task seeding
# ---------------------------------------------------------------------------

def copy_saved_tasks(*, src_dir: Path, tasks_root: Path) -> list[str]:
    """Copy provided saved task dirs (validate-*) into the workspace tasks root."""
    tasks_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for task_dir in sorted(src_dir.glob("validate-*")):
        if not task_dir.is_dir():
            continue
        dest = tasks_root / task_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(task_dir, dest)
        copied.append(task_dir.name)
    return copied


def synthesize_minimal_saved_task(*, tasks_root: Path, name: str = "validate-000001") -> Path:
    """A task dir that satisfies the static fill predicates but is not actually
    Docker-solvable; used only by the no-Docker smoke test.
    """
    task_dir = tasks_root / name
    task_subdir = task_dir / "task"
    (task_subdir / "original").mkdir(parents=True, exist_ok=True)
    (task_subdir / "reference").mkdir(parents=True, exist_ok=True)
    (task_subdir / "task.json").write_text(json.dumps({"name": name}) + "\n")
    (task_subdir / "task.txt").write_text("Dry-run placeholder issue.\n")
    (task_subdir / "commit.json").write_text(json.dumps({"sha": "0" * 40}) + "\n")
    # reference.patch must have >= 100 changed (+/-) lines.
    patch_lines = ["--- a/file.py", "+++ b/file.py", "@@ -0,0 +1,120 @@"]
    patch_lines += [f"+line {i}" for i in range(120)]
    (task_subdir / "reference.patch").write_text("\n".join(patch_lines) + "\n")
    (task_subdir / "original" / "file.py").write_text("")
    (task_subdir / "reference" / "file.py").write_text(
        "\n".join(f"line {i}" for i in range(120)) + "\n"
    )
    solutions = task_dir / "solutions" / "baseline"
    solutions.mkdir(parents=True, exist_ok=True)
    (solutions / "solution.diff").write_text("\n".join(patch_lines) + "\n")
    (solutions / "solve.json").write_text(
        json.dumps({"result": {"exit_reason": "completed", "elapsed_seconds": 120.0}}) + "\n"
    )
    return task_dir


# ---------------------------------------------------------------------------
# HF task-archive ingestion
# ---------------------------------------------------------------------------
# A tasks/ archive record stores a full task definition as base64 `artifacts`
# (the task/ tree, solutions/baseline, solutions/king, comparisons) plus a
# `pool_task` dict. Rollout records instead carry these keys and no artifacts.

_ROLLOUT_ONLY_KEYS = {"role", "final_patch", "trajectory"}


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def archive_jsonl_files(path: Path) -> list[Path]:
    """Collect *.jsonl / *.jsonl.gz under a file or directory."""
    if path.is_file():
        return [path]
    return sorted([*path.rglob("*.jsonl"), *path.rglob("*.jsonl.gz")])


def iter_archive_records(files: list[Path]) -> Iterator[tuple[dict[str, Any], str, Path]]:
    for f in files:
        pool_label = "retest" if "retest" in f.name else "primary"
        with _open_maybe_gzip(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line), pool_label, f


def classify_tasks_input(path: Path) -> str:
    """Return 'task_archive', 'rollouts', 'task_dirs', or 'empty'."""
    if path.is_dir() and any(p.is_dir() for p in path.glob("validate-*")):
        return "task_dirs"
    files = archive_jsonl_files(path)
    if not files:
        return "empty"
    for record, _label, _f in iter_archive_records(files[:1]):
        if "pool_task" in record and "artifacts" in record:
            return "task_archive"
        if _ROLLOUT_ONLY_KEYS & set(record):
            return "rollouts"
        return "unknown"
    return "empty"


def _write_artifact(dest_root: Path, artifact: dict[str, Any]) -> None:
    target = dest_root / artifact["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if artifact.get("encoding") == "base64":
        target.write_bytes(base64.b64decode(artifact["content_base64"]))
    else:
        target.write_text(artifact.get("content", ""), encoding="utf-8")


def expand_task_archive(
    *,
    files: list[Path],
    tasks_root: Path,
    validate_root: Path,
    target: int,
    king_hotkey: str,
) -> tuple[int, str | None]:
    """Expand up to *target* primary task records and pre-fill the pool.

    Each pool task is re-rooted at the workspace and relabeled to the seeded
    burn king, so the static gate passes on the archived king cache with no king
    Docker solve. Records whose king produced no patch (king_lines <= 0) are
    skipped, since the cache check requires matched_changed_lines > 0. Returns
    (seeded_count, king_commit_sha).
    """
    tasks_root.mkdir(parents=True, exist_ok=True)
    pool_dir = validate_root / "task-pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    seeded = 0
    king_commit: str | None = None
    for record, pool_label, _f in iter_archive_records(files):
        if seeded >= target:
            break
        if pool_label != "primary" or "pool_task" not in record or "artifacts" not in record:
            continue
        pool_task = dict(record["pool_task"])
        if int(pool_task.get("king_lines") or 0) <= 0:
            continue
        task_root_name = str(record["task_root_name"])
        dest = tasks_root / task_root_name
        if dest.exists():
            shutil.rmtree(dest)
        for artifact in record["artifacts"]:
            _write_artifact(dest, artifact)
        commit = str(pool_task.get("king_commit_sha") or "")
        king_commit = king_commit or commit
        pool_task["task_root"] = str(dest)
        pool_task["king_hotkey"] = king_hotkey
        (pool_dir / f"{task_root_name}.json").write_text(
            json.dumps(pool_task, indent=2, sort_keys=True) + "\n"
        )
        seeded += 1
    return seeded, king_commit


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_dry_run_config(
    *,
    workspace_root: Path,
    ninja_repo: Path,
    chain_snapshot: Path,
    wallet_name: str,
    wallet_hotkey: str,
    duel_rounds: int,
) -> RunConfig:
    return RunConfig(
        workspace_root=workspace_root,
        dry_run=True,
        validate_chain_mode="test",
        validate_chain_snapshot=chain_snapshot,
        validate_ninja_repo_local_path=ninja_repo,
        validate_wallet_name=wallet_name,
        validate_wallet_hotkey=wallet_hotkey,
        validate_private_submission_watch=True,
        validate_private_submission_only=True,
        validate_task_pool_fill_from_saved=True,
        validate_task_pool_static=True,
        # Target the round count; the caller caps it to the tasks actually seeded.
        validate_task_pool_target=duel_rounds,
        validate_duel_rounds=duel_rounds,
        validate_max_duels=1,
        validate_min_commitment_block=1,
        validate_hotkey_spent_since_block=1,
        validate_poll_interval_seconds=5,
        validate_max_concurrency=1,
        validate_round_concurrency=2,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _configure_external_env(*, r2_local_path: Path) -> None:
    # Drop live R2/HF/TMC credentials and upload flags, then point R2 at a local
    # directory so artifacts route through tau.io.r2.LocalS3Client, not the network.
    for var in (
        "R2_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
        "TMC_API_KEY", "HF_TOKEN",
        "VALIDATE_TASK_ARCHIVE_ENABLED", "TAU_PUSH_ROLLOUTS_TO_HF",
        "VALIDATE_TASK_ARCHIVE_HF_DATASET", "TAU_ROLLOUT_HF_DATASET",
    ):
        os.environ.pop(var, None)
    r2_local_path.mkdir(parents=True, exist_ok=True)
    os.environ["R2_LOCAL_PATH"] = str(r2_local_path)


def run_dry_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required for the real Docker solve + judge")

    ninja_repo = args.ninja_repo.expanduser().resolve()
    if not (ninja_repo / ".git").exists():
        raise SystemExit(f"--ninja-repo must be a git clone: {ninja_repo}")
    king_sha = git_head(ninja_repo)

    workspace = args.workspace.expanduser().resolve()
    config = build_dry_run_config(
        workspace_root=workspace,
        ninja_repo=ninja_repo,
        chain_snapshot=workspace / "chain-snapshot.json",
        wallet_name=args.wallet_name,
        wallet_hotkey=args.wallet_hotkey,
        duel_rounds=args.duel_rounds,
    )
    _configure_external_env(r2_local_path=workspace / "r2")

    challenger_hotkey = args.challenger_hotkey
    snapshot = synthesize_chain_snapshot(
        validator_hotkey=args.validator_hotkey,
        validator_uid=1,
        challenger_hotkey=challenger_hotkey,
        challenger_coldkey=args.challenger_coldkey,
        challenger_uid=2,
        registration_block=args.registration_block,
        block=args.block,
    )
    config.validate_chain_snapshot.parent.mkdir(parents=True, exist_ok=True)
    config.validate_chain_snapshot.write_text(json.dumps(snapshot, indent=2) + "\n")

    agent_text = (
        args.challenger_agent.expanduser().read_text() if args.challenger_agent else _DEFAULT_AGENT_PY
    )
    priv_root = config.validate_root / "private-submissions"
    seed_private_submission(
        root=priv_root,
        agent_py_text=agent_text,
        hotkey=challenger_hotkey,
        coldkey=args.challenger_coldkey,
        registration_block=args.registration_block,
    )

    # king_commit defaults to the ninja HEAD; the archive path overrides it to
    # the king its cached solution was produced against.
    king_commit = king_sha
    prefilled = False  # pool pre-filled and static-ready, no pool-manager needed
    available = 0      # distinct tasks seeded; caps the duel round count
    if args.tasks:
        tasks_path = args.tasks.expanduser().resolve()
        kind = classify_tasks_input(tasks_path)
        if kind == "rollouts":
            raise SystemExit(
                f"--tasks points at rollout recordings, not task definitions: {tasks_path}\n"
                "Rollouts (rollouts/<hour>/validate-*.jsonl) record king/challenger solutions and have\n"
                "no repo trees or reference patch, so they cannot seed a fresh-solve pool. Instead:\n"
                "  * fresh Docker duel: --tasks <ninja-rollouts/tasks/primary/<hour>_primary.jsonl>\n"
                "  * re-judge a recorded duel (just OpenRouter): python src/replay_duels.py ...\n"
            )
        if kind == "task_archive":
            files = archive_jsonl_files(tasks_path)
            seeded, archive_commit = expand_task_archive(
                files=files,
                tasks_root=config.tasks_root,
                validate_root=config.validate_root,
                target=config.validate_task_pool_target,
                king_hotkey=_BURN_KING_HOTKEY,
            )
            if seeded == 0:
                raise SystemExit(
                    f"no usable primary task records (with king_lines>0) found in {tasks_path}"
                )
            king_commit = archive_commit or king_sha
            available = seeded
            prefilled = True
            log.info(
                "Seeded %d task(s) from archive; pre-filled primary pool (king cache reused, "
                "no king Docker solve). king_commit=%s",
                seeded, king_commit[:12],
            )
        elif kind == "task_dirs":
            copied = copy_saved_tasks(src_dir=tasks_path, tasks_root=config.tasks_root)
            if not copied:
                raise SystemExit(f"no validate-* task dirs found under {tasks_path}")
            available = len(copied)
            log.info("Seeded %d saved task dir(s): %s", len(copied), ", ".join(copied))
        else:
            raise SystemExit(f"--tasks contains no usable task definitions ({kind}): {tasks_path}")
    else:
        synthesize_minimal_saved_task(tasks_root=config.tasks_root)
        available = 1
        log.warning(
            "No --tasks provided; synthesized a placeholder task that is NOT "
            "Docker-solvable. Pass --tasks with a tasks/ archive jsonl for a true run."
        )

    # A duel gathers duel_rounds distinct pool tasks; cap rounds and target to
    # the number seeded, or phase-1 gather blocks on tasks that never arrive.
    effective_rounds = max(1, min(args.duel_rounds, available))
    if effective_rounds != config.validate_duel_rounds:
        log.warning(
            "Capping duel rounds to %d (only %d task(s) seeded; requested %d)",
            effective_rounds, available, args.duel_rounds,
        )
    config = replace(
        config,
        validate_duel_rounds=effective_rounds,
        validate_task_pool_target=available,
    )

    seed_burn_king_state(validate_root=config.validate_root, ninja_repo=ninja_repo, sha=king_commit)

    bt.init(mode="test", snapshot=snapshot)

    import validate  # noqa: E402

    if prefilled:
        # Pool is king-matched and static-ready; only the challenger is solved.
        result = validate.validate_loop_run(config)
    else:
        # Task dirs may lack a king cache, so run the pool-manager alongside the
        # validator (as in production) to solve the king and fill the pool.
        import task_pool_manager  # noqa: E402

        def _pool_manager() -> None:
            try:
                task_pool_manager.run_pool_manager(replace(config, validate_max_duels=None))
            except Exception:  # pragma: no cover - background best-effort
                log.exception("pool-manager thread exited")

        threading.Thread(target=_pool_manager, name="pool-manager", daemon=True).start()
        result = validate.validate_loop_run(config)

    duels = sorted((config.validate_root / "duels").glob("*.json"))
    log.info("Dry-run finished: result=%s duels=%d", result, len(duels))
    if not duels:
        log.error("No duel artifact produced — check the pool filled and a challenger was queued")
        return 1
    print(json.dumps({"duels": [d.name for d in duels], "validate_root": str(config.validate_root)}, indent=2))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline dry-run of the validator (one duel).")
    parser.add_argument("--ninja-repo", type=Path, required=True, help="Local git clone of the miner ninja repo (king source).")
    parser.add_argument("--challenger-agent", type=Path, help="agent.py for the challenger private submission (default: minimal stub).")
    parser.add_argument("--tasks", type=Path, help="HF tasks/ archive jsonl (e.g. ninja-rollouts/tasks/primary/<hour>_primary.jsonl) OR a dir of saved task defs (validate-*). NOT a rollouts/ dir.")
    parser.add_argument("--workspace", type=Path, default=ROOT / "workspace-dryrun", help="Throwaway workspace root.")
    parser.add_argument("--wallet-name", default="dryrun")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--validator-hotkey", default="5DryRunValidatorHotkey0000000000000000000000000")
    parser.add_argument("--challenger-hotkey", default="5DryRunChallengerHotkey00000000000000000000000")
    parser.add_argument("--challenger-coldkey", default="5DryRunChallengerColdkey0000000000000000000000")
    parser.add_argument("--registration-block", type=int, default=2)
    parser.add_argument("--block", type=int, default=8_200_000)
    parser.add_argument("--duel-rounds", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_dry_run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
