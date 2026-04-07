from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx

from config import RunConfig, SolverAgentSource
from pipeline import _setup_logging, compare_task_run, generate_task_run, solve_task_run
from r2 import duel_to_summary, publish_dashboard_data
from workspace import write_json

log = logging.getLogger("swe-eval.validate")
_DEFAULT_GITHUB_AGENT_SUBDIR = "agent"
_GITHUB_COMMIT_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@(?P<sha>[0-9a-fA-F]{7,64})$"
)


@dataclass(slots=True)
class ValidatorSubmission:
    hotkey: str
    uid: int
    repo_full_name: str
    repo_url: str
    commit_sha: str
    commitment: str
    commitment_block: int
    local_path: str | None = None

    @property
    def agent_ref(self) -> str:
        return f"{self.repo_full_name}@{self.commit_sha}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidatorSubmission:
        return cls(
            hotkey=str(payload["hotkey"]),
            uid=int(payload["uid"]),
            repo_full_name=str(payload["repo_full_name"]),
            repo_url=str(payload["repo_url"]),
            commit_sha=str(payload["commit_sha"]),
            commitment=str(payload["commitment"]),
            commitment_block=int(payload["commitment_block"]),
            local_path=(
                str(payload["local_path"])
                if payload.get("local_path") is not None
                else None
            ),
        )


@dataclass(slots=True)
class ValidationRoundResult:
    task_name: str
    winner: str
    king_lines: int
    challenger_lines: int
    king_similarity_ratio: float
    challenger_similarity_ratio: float
    king_challenger_similarity: float
    task_root: str
    king_compare_root: str
    challenger_compare_root: str
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DuelResult:
    duel_id: int
    started_at: str
    finished_at: str
    king_before: ValidatorSubmission
    challenger: ValidatorSubmission
    rounds: list[ValidationRoundResult]
    wins: int
    losses: int
    ties: int
    king_after: ValidatorSubmission
    king_replaced: bool
    disqualification_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duel_id": self.duel_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "king_before": self.king_before.to_dict(),
            "challenger": self.challenger.to_dict(),
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "king_after": self.king_after.to_dict(),
            "king_replaced": self.king_replaced,
            "disqualification_reason": self.disqualification_reason,
        }


@dataclass(slots=True)
class ValidatorState:
    current_king: ValidatorSubmission | None = None
    queue: list[ValidatorSubmission] = field(default_factory=list)
    seen_hotkeys: list[str] = field(default_factory=list)
    retired_hotkeys: list[str] = field(default_factory=list)
    disqualified_hotkeys: list[str] = field(default_factory=list)
    locked_commitments: dict[str, str] = field(default_factory=dict)
    last_weight_block: int | None = None
    next_task_index: int = 1
    next_duel_index: int = 1
    king_since: str | None = None
    king_duels_defended: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_king": self.current_king.to_dict() if self.current_king else None,
            "queue": [submission.to_dict() for submission in self.queue],
            "seen_hotkeys": self.seen_hotkeys,
            "retired_hotkeys": self.retired_hotkeys,
            "disqualified_hotkeys": self.disqualified_hotkeys,
            "locked_commitments": self.locked_commitments,
            "last_weight_block": self.last_weight_block,
            "next_task_index": self.next_task_index,
            "next_duel_index": self.next_duel_index,
            "king_since": self.king_since,
            "king_duels_defended": self.king_duels_defended,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidatorState:
        current_king_payload = payload.get("current_king")
        raw_locked = payload.get("locked_commitments", {})
        return cls(
            current_king=(
                ValidatorSubmission.from_dict(current_king_payload)
                if isinstance(current_king_payload, dict)
                else None
            ),
            queue=[
                ValidatorSubmission.from_dict(item)
                for item in payload.get("queue", [])
                if isinstance(item, dict)
            ],
            seen_hotkeys=[str(item) for item in payload.get("seen_hotkeys", [])],
            retired_hotkeys=[str(item) for item in payload.get("retired_hotkeys", [])],
            disqualified_hotkeys=[str(item) for item in payload.get("disqualified_hotkeys", [])],
            locked_commitments={str(k): str(v) for k, v in raw_locked.items()} if isinstance(raw_locked, dict) else {},
            last_weight_block=(
                int(payload["last_weight_block"])
                if payload.get("last_weight_block") is not None
                else None
            ),
            next_task_index=int(payload.get("next_task_index", 1)),
            next_duel_index=int(payload.get("next_duel_index", 1)),
            king_since=payload.get("king_since"),
            king_duels_defended=int(payload.get("king_duels_defended", 0)),
        )


@dataclass(slots=True)
class ValidatePaths:
    root: Path
    state_path: Path
    duels_dir: Path


@dataclass(slots=True)
class ValidateStageResult:
    validate_root: str
    king_uid: int
    king_hotkey: str
    king_repo: str
    duel_count: int


@dataclass(slots=True)
class _MockNeuron:
    uid: int


class _MockCommitments:
    def get_all_revealed_commitments(self, netuid: int) -> dict[str, tuple[tuple[int, str], ...]]:
        _ = netuid
        return {}

    def get_all_commitments(self, netuid: int) -> dict[str, str]:
        _ = netuid
        return {}


class _MockSubnets:
    def __init__(self, hotkey_to_uid: dict[str, int]) -> None:
        self._hotkey_to_uid = hotkey_to_uid

    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int) -> int | None:
        _ = netuid
        return self._hotkey_to_uid.get(hotkey)


class _MockNeurons:
    def __init__(self, uids: list[int]) -> None:
        self._uids = uids

    def neurons_lite(self, netuid: int) -> list[_MockNeuron]:
        _ = netuid
        return [_MockNeuron(uid=uid) for uid in self._uids]


class _MockExtrinsics:
    def set_weights(self, **kwargs: Any) -> dict[str, Any]:
        return {"mocked": True, "called_with": kwargs}


class _MockSubtensorApi:
    def __init__(self) -> None:
        self._block = 1_000
        self._hotkey_to_uid = {
            "mock-king-hotkey": 1,
            "mock-challenger-hotkey": 2,
        }
        self.commitments = _MockCommitments()
        self.subnets = _MockSubnets(self._hotkey_to_uid)
        self.neurons = _MockNeurons(list(self._hotkey_to_uid.values()))
        self.extrinsics = _MockExtrinsics()

    @property
    def block(self) -> int:
        self._block += 1
        return self._block

    def __enter__(self) -> _MockSubtensorApi:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb


def _sprt_log_likelihood_ratio(wins: int, losses: int, p0: float, p1: float) -> float:
    if p0 <= 0 or p0 >= 1 or p1 <= 0 or p1 >= 1:
        raise ValueError("p0 and p1 must be in (0, 1)")
    return wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))


def _sprt_decision(
    wins: int,
    losses: int,
    epsilon: float,
    alpha: float,
    beta: float,
) -> str:
    """Return 'challenger' if contender is statistically better, 'king' if not, 'continue' if undecided."""
    p0 = 0.5
    p1 = min(0.5 + epsilon, 0.999)
    lower_bound = math.log(beta / (1 - alpha))
    upper_bound = math.log((1 - beta) / alpha)
    llr = _sprt_log_likelihood_ratio(wins, losses, p0, p1)
    if llr >= upper_bound:
        return "challenger"
    if llr <= lower_bound:
        return "king"
    return "continue"


def validate_loop_run(config: RunConfig) -> ValidateStageResult:
    _setup_logging(debug=config.debug)
    if config.validate_rounds < 1:
        raise ValueError("--rounds must be at least 1")
    if config.validate_concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if config.validate_eval_window_seconds < 1:
        raise ValueError("--eval-window-seconds must be at least 1")
    if config.validate_weight_interval_blocks < 1:
        raise ValueError("--weight-interval-blocks must be at least 1")
    if config.validate_max_duels is not None and config.validate_max_duels < 1:
        raise ValueError("--max-duels must be at least 1")
    if config.validate_epsilon <= 0 or config.validate_epsilon >= 0.5:
        raise ValueError("--epsilon must be in (0, 0.5)")
    if config.validate_alpha <= 0 or config.validate_alpha >= 1:
        raise ValueError("--alpha must be in (0, 1)")
    if config.validate_beta <= 0 or config.validate_beta >= 1:
        raise ValueError("--beta must be in (0, 1)")
    if config.validate_min_rounds < 1:
        raise ValueError("--min-rounds must be at least 1")
    if config.validate_max_rounds < config.validate_min_rounds:
        raise ValueError("--max-rounds must be >= --min-rounds")
    if (
        not config.validate_mock_set_weights
        and (not config.validate_wallet_name or not config.validate_wallet_hotkey)
    ):
        raise ValueError("validate requires --wallet-name and --wallet-hotkey")

    paths = _prepare_validate_paths(config.validate_root)
    state = _load_state(paths.state_path)
    dashboard_history = _load_dashboard_history(paths.root / "dashboard_history.json")
    if dashboard_history:
        max_duel_id = max(d.get("duel_id", 0) for d in dashboard_history)
        if max_duel_id >= state.next_duel_index:
            state.next_duel_index = max_duel_id + 1
        max_task_idx = 0
        for d in dashboard_history:
            for r in d.get("rounds", []):
                tn = r.get("task_name", "")
                parts = tn.rsplit("-", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_task_idx = max(max_task_idx, int(parts[1]))
        if max_task_idx >= state.next_task_index:
            state.next_task_index = max_task_idx + 1
    validator_started_at = _timestamp()
    active_duel_info: dict[str, Any] | None = None
    github_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "swe-eval-validate",
    }
    if config.github_token:
        github_headers["Authorization"] = f"Bearer {config.github_token}"
    github_client = httpx.Client(
        base_url="https://api.github.com",
        headers=github_headers,
        follow_redirects=True,
        timeout=config.http_timeout,
    )
    duel_count = 0

    try:
        with _open_subtensor(config) as subtensor:
            log.info("Validator connected to chain; starting main loop for netuid %s", config.validate_netuid)
            while True:
                current_block = subtensor.block
                log.info("Poll cycle block=%s king=%s queue=%d seen=%d",
                         current_block,
                         state.current_king.commitment if state.current_king else None,
                         len(state.queue),
                         len(state.seen_hotkeys))
                chain_submissions = _fetch_chain_submissions(
                    subtensor=subtensor, github_client=github_client, config=config
                )
                _refresh_queue(chain_submissions=chain_submissions, config=config, state=state)
                prev_king_hotkey = state.current_king.hotkey if state.current_king else None
                _ensure_king(state=state)
                if state.current_king and state.current_king.hotkey != prev_king_hotkey:
                    state.king_since = _timestamp()
                    state.king_duels_defended = 0

                # Ensure king SHA is resolved to full 40-char form
                # (may be short if loaded from state written by older code)
                if state.current_king and len(state.current_king.commit_sha) < 40:
                    full = _resolve_public_commit(
                        github_client, state.current_king.repo_full_name, state.current_king.commit_sha
                    )
                    if full:
                        state.current_king.commit_sha = full

                if state.current_king is None:
                    log.info("No valid king or challengers found on subnet %s yet; sleeping", config.validate_netuid)
                    _save_state(paths.state_path, state)
                    _publish_dashboard(state, dashboard_history, config, validator_started_at, None)
                    time.sleep(config.validate_poll_interval_seconds)
                    continue

                maybe_promoted = _maybe_disqualify_king(
                    subtensor=subtensor,
                    github_client=github_client,
                    config=config,
                    state=state,
                    reason_prefix="Current king is no longer eligible",
                )
                if maybe_promoted:
                    current_block = subtensor.block

                _maybe_set_weights(
                    subtensor=subtensor,
                    config=config,
                    state=state,
                    current_block=current_block,
                )

                challenger = _pop_next_valid_challenger(
                    subtensor=subtensor,
                    github_client=github_client,
                    config=config,
                    state=state,
                )
                if challenger is None:
                    log.info("No challenger in queue; sleeping %ds", config.validate_poll_interval_seconds)
                    _save_state(paths.state_path, state)
                    _publish_dashboard(state, dashboard_history, config, validator_started_at, None)
                    time.sleep(config.validate_poll_interval_seconds)
                    continue

                log.info("Starting duel: king=%s vs challenger=%s (uid %s)",
                         state.current_king.commitment if state.current_king else "?",
                         challenger.commitment, challenger.uid)
                active_duel_info = {
                    "duel_id": state.next_duel_index,
                    "king_uid": state.current_king.uid if state.current_king else None,
                    "king_repo": state.current_king.repo_full_name if state.current_king else None,
                    "challenger_uid": challenger.uid,
                    "challenger_repo": challenger.repo_full_name,
                    "started_at": _timestamp(),
                    "rounds_completed": 0,
                    "wins": 0, "losses": 0, "ties": 0,
                }
                _publish_dashboard(state, dashboard_history, config, validator_started_at, active_duel_info)

                duel = _run_duel(
                    subtensor=subtensor,
                    github_client=github_client,
                    config=config,
                    state=state,
                    challenger=challenger,
                )
                active_duel_info = None
                duel_count += 1

                if duel.king_replaced:
                    state.king_since = _timestamp()
                    state.king_duels_defended = 0
                else:
                    state.king_duels_defended += 1

                log.info("Duel %d complete: wins=%d losses=%d ties=%d king_replaced=%s",
                         duel_count, duel.wins, duel.losses, duel.ties, duel.king_replaced)
                _write_duel(paths, duel)
                _save_state(paths.state_path, state)

                dashboard_history.append(duel_to_summary(duel.to_dict()))
                _save_dashboard_history(paths.root / "dashboard_history.json", dashboard_history)
                _publish_dashboard(state, dashboard_history, config, validator_started_at, None)

                if config.validate_max_duels is not None and duel_count >= config.validate_max_duels:
                    break
    finally:
        github_client.close()

    current_king = state.current_king
    if current_king is None:
        raise RuntimeError("validate loop exited without a current king")
    return ValidateStageResult(
        validate_root=str(paths.root),
        king_uid=current_king.uid,
        king_hotkey=current_king.hotkey,
        king_repo=current_king.agent_ref,
        duel_count=duel_count,
    )


def _run_duel(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    state: ValidatorState,
    challenger: ValidatorSubmission,
) -> DuelResult:
    if state.current_king is None:
        raise RuntimeError("Cannot start duel without a king")

    king_before = state.current_king
    duel_id = state.next_duel_index
    state.next_duel_index += 1
    started_at = _timestamp()
    deadline = time.monotonic() + config.validate_eval_window_seconds
    rounds: list[ValidationRoundResult] = []
    wins = 0
    losses = 0
    ties = 0
    launched = 0
    scored = 0
    sprt_verdict = "continue"

    log.info(
        "Starting duel %s: king uid=%s (%s) vs challenger uid=%s (%s)",
        duel_id,
        king_before.uid,
        king_before.agent_ref,
        challenger.uid,
        challenger.agent_ref,
    )

    if not _submission_is_eligible(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        submission=challenger,
    ):
        _mark_disqualified(state, challenger.hotkey)
        finished_at = _timestamp()
        return DuelResult(
            duel_id=duel_id,
            started_at=started_at,
            finished_at=finished_at,
            king_before=king_before,
            challenger=challenger,
            rounds=[],
            wins=0,
            losses=0,
            ties=0,
            king_after=king_before,
            king_replaced=False,
            disqualification_reason="challenger is not eligible",
        )

    with ThreadPoolExecutor(max_workers=config.validate_concurrency) as executor:
        futures: dict[Future[ValidationRoundResult], str] = {}

        while scored < config.validate_max_rounds and sprt_verdict == "continue":
            while (
                len(futures) < config.validate_concurrency
                and launched < config.validate_max_rounds
                and time.monotonic() < deadline
                and sprt_verdict == "continue"
            ):
                task_name = _allocate_task_name(state)
                future = executor.submit(
                    _run_validation_round,
                    task_name=task_name,
                    duel_id=duel_id,
                    king=king_before,
                    challenger=challenger,
                    config=config,
                )
                futures[future] = task_name
                launched += 1

            if not futures:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            done, _ = wait(
                futures,
                timeout=min(remaining, 1.0),
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue

            for future in done:
                result = future.result()
                futures.pop(future, None)
                rounds.append(result)
                if result.scored:
                    scored += 1
                    if result.winner == "challenger":
                        wins += 1
                    elif result.winner == "king":
                        losses += 1
                    else:
                        ties += 1

                    decisive = wins + losses
                    if decisive > 0 and scored >= config.validate_min_rounds:
                        sprt_verdict = _sprt_decision(
                            wins=wins,
                            losses=losses,
                            epsilon=config.validate_epsilon,
                            alpha=config.validate_alpha,
                            beta=config.validate_beta,
                        )
                        log.info(
                            "Duel %s SPRT after %d scored (%d decisive): "
                            "W=%d L=%d T=%d verdict=%s",
                            duel_id, scored, decisive, wins, losses, ties, sprt_verdict,
                        )

    if scored >= config.validate_max_rounds and sprt_verdict == "continue":
        sprt_verdict = "king"
        log.info("Duel %s hit max rounds (%d); defaulting to king", duel_id, config.validate_max_rounds)

    disqualification_reason: str | None = None
    king_after = king_before
    king_replaced = False

    if not _submission_is_eligible(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        submission=king_before,
    ):
        _mark_disqualified(state, king_before.hotkey)
        replacement = _resolve_promotion_candidate(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            state=state,
            primary_candidate=challenger,
        )
        if replacement is not None:
            king_after = replacement
            king_replaced = True
        disqualification_reason = "king is no longer eligible"
    elif sprt_verdict == "challenger":
        scored_with_similarity = [
            r for r in rounds if r.scored and r.king_challenger_similarity > 0
        ]
        if scored_with_similarity:
            mean_kc_sim = sum(r.king_challenger_similarity for r in scored_with_similarity) / len(
                scored_with_similarity
            )
        else:
            mean_kc_sim = 0.0

        if mean_kc_sim >= config.validate_copy_similarity_threshold:
            _mark_disqualified(state, challenger.hotkey)
            disqualification_reason = (
                f"challenger disqualified as likely copy "
                f"(mean king-challenger similarity {mean_kc_sim:.3f} "
                f">= threshold {config.validate_copy_similarity_threshold})"
            )
            log.warning("Duel %s: %s", duel_id, disqualification_reason)
        else:
            replacement = _resolve_promotion_candidate(
                subtensor=subtensor,
                github_client=github_client,
                config=config,
                state=state,
                primary_candidate=challenger,
            )
            if replacement is not None:
                _retire_hotkey(state, king_before.hotkey)
                king_after = replacement
                king_replaced = True

    if disqualification_reason is not None and not king_replaced:
        state.current_king = None
    else:
        state.current_king = king_after
    finished_at = _timestamp()
    king_label = state.current_king.agent_ref if state.current_king is not None else "<none>"
    log.info(
        "Finished duel %s: wins=%s losses=%s ties=%s verdict=%s king=%s",
        duel_id,
        wins,
        losses,
        ties,
        sprt_verdict,
        king_label,
    )
    return DuelResult(
        duel_id=duel_id,
        started_at=started_at,
        finished_at=finished_at,
        king_before=king_before,
        challenger=challenger,
        rounds=rounds,
        wins=wins,
        losses=losses,
        ties=ties,
        king_after=king_after,
        king_replaced=king_replaced,
        disqualification_reason=disqualification_reason,
    )


def _run_validation_round(
    *,
    task_name: str,
    duel_id: int,
    king: ValidatorSubmission,
    challenger: ValidatorSubmission,
    config: RunConfig,
) -> ValidationRoundResult:
    if config.validate_mock_rounds:
        return _run_mock_validation_round(
            task_name=task_name,
            duel_id=duel_id,
            king=king,
            challenger=challenger,
            config=config,
        )

    try:
        generate_result = generate_task_run(task_name=task_name, config=config)

        cursor_start = time.monotonic()
        cursor_result = solve_task_run(
            task_name=task_name,
            solution_name="cursor",
            config=_build_cursor_config(config),
        )
        cursor_elapsed = time.monotonic() - cursor_start

        _AGENT_TIMEOUT_FLOOR = 300
        agent_timeout = max(int(cursor_elapsed * 2) + 1, _AGENT_TIMEOUT_FLOOR)
        log.info(
            "Cursor solved %s in %.1fs; agent timeout capped to %ds",
            task_name, cursor_elapsed, agent_timeout,
        )

        king_cfg = replace(_build_agent_config(config, king), agent_timeout=agent_timeout)
        king_start = time.monotonic()
        _king_result = solve_task_run(
            task_name=task_name,
            solution_name="king",
            config=king_cfg,
        )
        king_elapsed = time.monotonic() - king_start
        king_timed_out = king_elapsed >= agent_timeout

        challenger_cfg = replace(_build_agent_config(config, challenger), agent_timeout=agent_timeout)
        challenger_start = time.monotonic()
        _challenger_result = solve_task_run(
            task_name=task_name,
            solution_name="challenger",
            config=challenger_cfg,
        )
        challenger_elapsed = time.monotonic() - challenger_start
        challenger_timed_out = challenger_elapsed >= agent_timeout

        if king_timed_out:
            log.info("King timed out on %s (%.1fs >= %ds); zero score", task_name, king_elapsed, agent_timeout)
        if challenger_timed_out:
            log.info("Challenger timed out on %s (%.1fs >= %ds); zero score", task_name, challenger_elapsed, agent_timeout)

        king_compare = compare_task_run(
            task_name=task_name,
            solution_names=["cursor", "king"],
            config=config,
        )
        challenger_compare = compare_task_run(
            task_name=task_name,
            solution_names=["cursor", "challenger"],
            config=config,
        )
        king_challenger_compare = compare_task_run(
            task_name=task_name,
            solution_names=["king", "challenger"],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return ValidationRoundResult(
            task_name=task_name,
            winner="error",
            king_lines=0,
            challenger_lines=0,
            king_similarity_ratio=0.0,
            challenger_similarity_ratio=0.0,
            king_challenger_similarity=0.0,
            task_root=str(config.tasks_root / task_name),
            king_compare_root="",
            challenger_compare_root="",
            error=f"duel {duel_id} task {task_name} failed: {exc}",
        )

    _ = cursor_result

    king_lines = 0 if king_timed_out else king_compare.matched_changed_lines
    challenger_lines = 0 if challenger_timed_out else challenger_compare.matched_changed_lines

    if challenger_lines > king_lines:
        winner = "challenger"
    elif challenger_lines < king_lines:
        winner = "king"
    else:
        winner = "tie"

    return ValidationRoundResult(
        task_name=task_name,
        winner=winner,
        king_lines=king_lines,
        challenger_lines=challenger_lines,
        king_similarity_ratio=0.0 if king_timed_out else king_compare.similarity_ratio,
        challenger_similarity_ratio=0.0 if challenger_timed_out else challenger_compare.similarity_ratio,
        king_challenger_similarity=king_challenger_compare.similarity_ratio,
        task_root=generate_result.task_root,
        king_compare_root=king_compare.comparison_root,
        challenger_compare_root=challenger_compare.comparison_root,
    )


def _run_mock_validation_round(
    *,
    task_name: str,
    duel_id: int,
    king: ValidatorSubmission,
    challenger: ValidatorSubmission,
    config: RunConfig,
) -> ValidationRoundResult:
    task_root = config.tasks_root / task_name
    king_compare_root = task_root / "comparisons" / "cursor--vs--king"
    challenger_compare_root = task_root / "comparisons" / "cursor--vs--challenger"
    challenger_lines = max(king.uid, challenger.uid) + 9
    king_lines = min(king.uid, challenger.uid) + 4
    winner = "challenger" if challenger_lines > king_lines else "tie"
    king_compare_root.mkdir(parents=True, exist_ok=True)
    challenger_compare_root.mkdir(parents=True, exist_ok=True)
    log.info(
        "Mock round for duel %s task %s using local agent %s: king uid=%s challenger uid=%s winner=%s",
        duel_id,
        task_name,
        config.validate_mock_local_agent,
        king.uid,
        challenger.uid,
        winner,
    )
    return ValidationRoundResult(
        task_name=task_name,
        winner=winner,
        king_lines=king_lines,
        challenger_lines=challenger_lines,
        king_similarity_ratio=0.40,
        challenger_similarity_ratio=0.75,
        king_challenger_similarity=0.30,
        task_root=str(task_root),
        king_compare_root=str(king_compare_root),
        challenger_compare_root=str(challenger_compare_root),
    )


def _refresh_queue(
    *,
    chain_submissions: list[ValidatorSubmission],
    config: RunConfig,
    state: ValidatorState,
) -> None:
    known_hotkeys = set(state.seen_hotkeys)
    if state.current_king:
        known_hotkeys.add(state.current_king.hotkey)
    known_hotkeys.update(submission.hotkey for submission in state.queue)
    submissions = chain_submissions
    queue_limit = config.validate_queue_size
    for submission in submissions:
        locked = state.locked_commitments.get(submission.hotkey)
        if locked is not None and locked != submission.commitment:
            log.warning(
                "Hotkey %s changed commitment from %r to %r; ignoring (immutable)",
                submission.hotkey,
                locked,
                submission.commitment,
            )
            continue
        if submission.hotkey in known_hotkeys:
            continue
        if queue_limit is not None and len(state.queue) >= queue_limit:
            break
        state.locked_commitments[submission.hotkey] = submission.commitment
        state.queue.append(submission)
        state.seen_hotkeys.append(submission.hotkey)
        known_hotkeys.add(submission.hotkey)

    state.queue.sort(key=lambda item: (item.commitment_block, item.uid, item.hotkey))


def _fetch_chain_submissions(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
) -> list[ValidatorSubmission]:
    if config.validate_mock_local_agent:
        _ = github_client
        return _mock_submissions(subtensor=subtensor, config=config)

    revealed = subtensor.commitments.get_all_revealed_commitments(config.validate_netuid)
    current_commitments = subtensor.commitments.get_all_commitments(config.validate_netuid)
    submissions: list[ValidatorSubmission] = []
    seen_hotkeys: set[str] = set()
    current_block = subtensor.block

    for hotkey, entries in revealed.items():
        normalized_entries: list[tuple[int, str]] = []
        if isinstance(entries, tuple):
            for item in entries:
                if not isinstance(item, tuple) or len(item) != 2:
                    continue
                normalized_entries.append((int(item[0]), str(item[1])))
        if not normalized_entries:
            continue
        earliest_block, commitment = min(normalized_entries, key=lambda item: item[0])
        submission = _build_submission(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            hotkey=str(hotkey),
            commitment=str(commitment),
            commitment_block=int(earliest_block),
        )
        if submission is not None:
            submissions.append(submission)
            seen_hotkeys.add(submission.hotkey)

    for hotkey, commitment in current_commitments.items():
        hotkey = str(hotkey)
        if hotkey in seen_hotkeys:
            continue
        submission = _build_submission(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            hotkey=hotkey,
            commitment=str(commitment),
            commitment_block=current_block,
        )
        if submission is not None:
            submissions.append(submission)

    submissions.sort(key=lambda item: (item.commitment_block, item.uid, item.hotkey))
    return submissions


def _mock_submissions(*, subtensor, config: RunConfig) -> list[ValidatorSubmission]:
    if not config.validate_mock_local_agent:
        return []
    local_agent_path = str(Path(config.validate_mock_local_agent).expanduser().resolve())
    block = subtensor.block
    submissions = [
        ValidatorSubmission(
            hotkey="mock-king-hotkey",
            uid=1,
            repo_full_name="local/mock-agent",
            repo_url=local_agent_path,
            commit_sha="local-king",
            commitment="local/mock-agent@local-king",
            commitment_block=block - 1,
            local_path=local_agent_path,
        ),
        ValidatorSubmission(
            hotkey="mock-challenger-hotkey",
            uid=2,
            repo_full_name="local/mock-agent",
            repo_url=local_agent_path,
            commit_sha="local-challenger",
            commitment="local/mock-agent@local-challenger",
            commitment_block=block,
            local_path=local_agent_path,
        ),
    ]
    log.info(
        "Using %s mock submissions backed by local agent %s",
        len(submissions),
        local_agent_path,
    )
    return submissions


def _build_submission(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    hotkey: str,
    commitment: str,
    commitment_block: int,
) -> ValidatorSubmission | None:
    parsed = _parse_submission_commitment(commitment)
    if parsed is None:
        log.warning("Skipping malformed commitment for hotkey %s: %r", hotkey, commitment)
        return None

    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(hotkey, config.validate_netuid)
    if uid is None:
        log.warning("Skipping commitment for unregistered hotkey %s", hotkey)
        return None

    repo_full_name, commit_sha = parsed
    full_sha = _resolve_public_commit(github_client, repo_full_name, commit_sha)
    if full_sha is None:
        log.warning("Skipping non-public submission for hotkey %s: %s@%s", hotkey, repo_full_name, commit_sha)
        return None

    return ValidatorSubmission(
        hotkey=hotkey,
        uid=int(uid),
        repo_full_name=repo_full_name,
        repo_url=f"https://github.com/{repo_full_name}.git",
        commit_sha=full_sha,
        commitment=commitment,
        commitment_block=commitment_block,
    )


def _ensure_king(*, state: ValidatorState) -> None:
    if state.current_king is not None:
        return
    if not state.queue:
        return
    state.current_king = state.queue.pop(0)


def _pop_next_valid_challenger(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    state: ValidatorState,
) -> ValidatorSubmission | None:
    while state.queue:
        candidate = state.queue.pop(0)
        if _submission_is_eligible(
            subtensor=subtensor,
            github_client=github_client,
            config=config,
            submission=candidate,
        ):
            return candidate
        _mark_disqualified(state, candidate.hotkey)
    return None


def _submission_is_eligible(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    submission: ValidatorSubmission,
) -> bool:
    current_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(submission.hotkey, config.validate_netuid)
    if current_uid is None:
        return False
    if submission.local_path is not None:
        submission.uid = int(current_uid)
        return True
    if not _is_public_commit(github_client, submission.repo_full_name, submission.commit_sha):
        return False
    submission.uid = int(current_uid)
    return True


def _maybe_disqualify_king(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    state: ValidatorState,
    reason_prefix: str,
) -> bool:
    king = state.current_king
    if king is None:
        return False
    if _submission_is_eligible(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        submission=king,
    ):
        return False

    _mark_disqualified(state, king.hotkey)
    state.current_king = None
    state.current_king = _pop_next_valid_challenger(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        state=state,
    )
    if state.current_king is None:
        log.warning("%s, and no replacement is queued", reason_prefix)
        return True

    log.warning(
        "%s; promoted queued challenger uid=%s (%s)",
        reason_prefix,
        state.current_king.uid,
        state.current_king.agent_ref,
    )
    return True


def _maybe_set_weights(*, subtensor, config: RunConfig, state: ValidatorState, current_block: int) -> None:
    king = state.current_king
    if king is None:
        return
    last_weight_block = state.last_weight_block
    if last_weight_block is not None and current_block - last_weight_block < config.validate_weight_interval_blocks:
        return

    neurons = list(subtensor.neurons.neurons_lite(config.validate_netuid))
    if not neurons:
        raise RuntimeError(f"Subnet {config.validate_netuid} has no neurons")

    current_uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(king.hotkey, config.validate_netuid)
    if current_uid is None:
        raise RuntimeError(f"Current king {king.hotkey} is no longer registered")

    king.uid = int(current_uid)
    uids = [int(neuron.uid) for neuron in neurons]
    weights = [1.0 if uid == king.uid else 0.0 for uid in uids]
    if config.validate_mock_set_weights:
        state.last_weight_block = current_block
        log.info(
            "Mocked set_weights at block %s for netuid %s to king uid=%s uids=%s weights=%s",
            current_block,
            config.validate_netuid,
            king.uid,
            uids,
            weights,
        )
        return
    wallet = bt.Wallet(
        name=config.validate_wallet_name,
        hotkey=config.validate_wallet_hotkey,
        path=config.validate_wallet_path,
    )
    response = subtensor.extrinsics.set_weights(
        wallet=wallet,
        netuid=config.validate_netuid,
        uids=uids,
        weights=weights,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    state.last_weight_block = current_block
    log.info(
        "Set weights at block %s for netuid %s to king uid=%s response=%s",
        current_block,
        config.validate_netuid,
        king.uid,
        response,
    )


def _build_cursor_config(config: RunConfig) -> RunConfig:
    return replace(
        config,
        solver_backend="cursor",
        solve_agent="cursor",
        solver_agent_source=None,
    )


def _build_agent_config(config: RunConfig, submission: ValidatorSubmission) -> RunConfig:
    if submission.local_path is not None:
        agent_source = SolverAgentSource(
            raw=submission.local_path,
            kind="local_path",
            local_path=submission.local_path,
        )
        return replace(
            config,
            solver_backend="docker-pi",
            solve_agent=submission.local_path,
            solver_agent_source=agent_source,
        )
    agent_source = SolverAgentSource(
        raw=submission.agent_ref,
        kind="github_repo",
        repo_url=submission.repo_url,
        agent_subdir=_DEFAULT_GITHUB_AGENT_SUBDIR,
        commit_sha=submission.commit_sha,
    )
    return replace(
        config,
        solver_backend="docker-pi",
        solve_agent=submission.agent_ref,
        solver_agent_source=agent_source,
    )


def _allocate_task_name(state: ValidatorState) -> str:
    index = state.next_task_index
    state.next_task_index += 1
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"validate-{timestamp}-{index:06d}"


def _prepare_validate_paths(root: Path) -> ValidatePaths:
    root.mkdir(parents=True, exist_ok=True)
    duels_dir = root / "duels"
    duels_dir.mkdir(parents=True, exist_ok=True)
    return ValidatePaths(
        root=root,
        state_path=root / "state.json",
        duels_dir=duels_dir,
    )


def _load_state(path: Path) -> ValidatorState:
    if not path.exists():
        return ValidatorState()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid validator state file: {path}")
    return ValidatorState.from_dict(payload)


def _save_state(path: Path, state: ValidatorState) -> None:
    write_json(path, state.to_dict())


def _write_duel(paths: ValidatePaths, duel: DuelResult) -> None:
    duel_path = paths.duels_dir / f"{duel.duel_id:06d}.json"
    write_json(duel_path, duel.to_dict())


def _load_dashboard_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            return payload
    except Exception:
        log.exception("Failed to load dashboard history from %s; starting fresh", path)
    return []


def _save_dashboard_history(path: Path, history: list[dict[str, Any]]) -> None:
    write_json(path, history)


def _publish_dashboard(
    state: ValidatorState,
    history: list[dict[str, Any]],
    config: RunConfig,
    validator_started_at: str,
    active_duel: dict[str, Any] | None,
) -> None:
    king = state.current_king
    king_dict = {
        "uid": king.uid,
        "hotkey": king.hotkey,
        "repo_full_name": king.repo_full_name,
        "repo_url": f"https://github.com/{king.repo_full_name}",
        "commit_sha": king.commit_sha,
    } if king else None

    commitment_to_submission: dict[str, dict[str, Any]] = {}
    for d in history:
        for role in ("king", "challenger"):
            hk = d.get(f"{role}_hotkey")
            if hk and hk not in commitment_to_submission:
                commitment_to_submission[hk] = {
                    "uid": d.get(f"{role}_uid"),
                    "hotkey": hk,
                    "repo": d.get(f"{role}_repo"),
                }

    def _resolve_hotkey(hk: str) -> dict[str, Any]:
        if hk in commitment_to_submission:
            return commitment_to_submission[hk]
        commitment = state.locked_commitments.get(hk, "")
        repo = commitment.split("@")[0] if "@" in commitment else commitment
        return {"uid": None, "hotkey": hk, "repo": repo or "unknown"}

    total_rounds = sum(len(d.get("rounds", [])) for d in history)
    status = {
        "validator_started_at": validator_started_at,
        "netuid": config.validate_netuid,
        "queue": [
            {"uid": s.uid, "repo": s.repo_full_name, "hotkey": s.hotkey, "commitment_block": s.commitment_block}
            for s in state.queue
        ],
        "active_duel": active_duel,
        "disqualified": [_resolve_hotkey(hk) for hk in state.disqualified_hotkeys],
        "retired": [_resolve_hotkey(hk) for hk in state.retired_hotkeys],
        "total_rounds": total_rounds,
        "miners_seen": len(state.seen_hotkeys),
        "king_since": state.king_since,
        "king_duels_defended": state.king_duels_defended,
    }

    local_payload = {
        "updated_at": _timestamp(),
        "current_king": king_dict,
        "duels": history,
        "status": status,
    }
    local_path = config.validate_root / "dashboard_data.json"
    try:
        write_json(local_path, local_payload)
    except Exception:
        log.exception("Failed to write local dashboard_data.json (non-fatal)")

    try:
        publish_dashboard_data(current_king=king_dict, duel_history=history, status=status)
    except Exception:
        log.exception("R2 dashboard publish failed (non-fatal)")


def _retire_hotkey(state: ValidatorState, hotkey: str) -> None:
    if hotkey not in state.retired_hotkeys:
        state.retired_hotkeys.append(hotkey)


def _mark_disqualified(state: ValidatorState, hotkey: str) -> None:
    if hotkey not in state.disqualified_hotkeys:
        state.disqualified_hotkeys.append(hotkey)


def _resolve_promotion_candidate(
    *,
    subtensor,
    github_client: httpx.Client,
    config: RunConfig,
    state: ValidatorState,
    primary_candidate: ValidatorSubmission,
) -> ValidatorSubmission | None:
    if _submission_is_eligible(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        submission=primary_candidate,
    ):
        return primary_candidate

    _mark_disqualified(state, primary_candidate.hotkey)
    return _pop_next_valid_challenger(
        subtensor=subtensor,
        github_client=github_client,
        config=config,
        state=state,
    )


def _parse_submission_commitment(raw_value: str) -> tuple[str, str] | None:
    cleaned = raw_value.strip().rstrip("/")
    match = _GITHUB_COMMIT_RE.fullmatch(cleaned)
    if match:
        return match.group("repo"), match.group("sha")

    prefix = "https://github.com/"
    if cleaned.startswith(prefix):
        path = cleaned[len(prefix) :]
    elif cleaned.startswith("github.com/"):
        path = cleaned[len("github.com/") :]
    else:
        return None

    parts = [part for part in path.split("/") if part]
    if len(parts) >= 4 and parts[2] == "commit":
        repo_full_name = "/".join(parts[:2])
        return repo_full_name, parts[3]
    return None


def _resolve_public_commit(github_client: httpx.Client, repo_full_name: str, commit_sha: str) -> str | None:
    """Return the full 40-char SHA if the commit exists in a public repo, else None."""
    repo_response = github_client.get(f"/repos/{repo_full_name}")
    if repo_response.status_code != 200:
        return None
    repo_payload = repo_response.json()
    if repo_payload.get("private") is not False:
        return None

    commit_response = github_client.get(f"/repos/{repo_full_name}/commits/{commit_sha}")
    if commit_response.status_code != 200:
        return None
    return commit_response.json().get("sha", commit_sha)


def _is_public_commit(github_client: httpx.Client, repo_full_name: str, commit_sha: str) -> bool:
    return _resolve_public_commit(github_client, repo_full_name, commit_sha) is not None


def _open_subtensor(config: RunConfig):
    if config.validate_mock_local_agent:
        return _MockSubtensorApi()
    network = config.validate_subtensor_endpoint or config.validate_network
    # Disable the default 5-second websocket auto-shutdown so the connection
    # survives idle periods (e.g. while resolving GitHub SHAs between chain queries).
    if network:
        return bt.SubtensorApi(network=network, websocket_shutdown_timer=0)
    return bt.SubtensorApi(websocket_shutdown_timer=0)


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()
