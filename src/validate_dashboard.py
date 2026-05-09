from __future__ import annotations

import logging
import os
from typing import Any, Callable

from r2 import publish_dashboard_data
from workspace import write_json

log = logging.getLogger("swe-eval.validate")


def _publish_dashboard(
    state: Any,
    history: list[dict[str, Any]],
    config: Any,
    validator_started_at: str,
    active_duel: dict[str, Any] | None = None,
    chain_data: dict[str, Any] | None = None,
    *,
    dashboard_links_fn: Callable[[], dict[str, str]],
    dashboard_submission_dict_fn: Callable[..., dict[str, Any]],
    effective_recent_kings_fn: Callable[[Any], list[Any]],
    diff_judge_weight: float,
    resolve_diff_judge_models_fn: Callable[[Any], tuple[str, str]],
    timestamp_fn: Callable[[], str],
) -> None:
    king = state.current_king
    king_dict = dashboard_submission_dict_fn(king, history=history) if king else None

    active_duel_info = active_duel
    links = dashboard_links_fn()

    commitment_map: dict[str, dict[str, Any]] = {}
    for d in history:
        for role in ("king", "challenger"):
            hk = d.get(f"{role}_hotkey")
            if hk and hk not in commitment_map:
                commitment_map[hk] = {"uid": d.get(f"{role}_uid"), "hotkey": hk, "repo": d.get(f"{role}_repo")}

    def _resolve_hk(hk: str) -> dict[str, Any]:
        if hk in commitment_map:
            return commitment_map[hk]
        c = state.locked_commitments.get(hk, "")
        repo = c.split("@")[0] if "@" in c else c
        return {"uid": None, "hotkey": hk, "repo": repo or "unknown"}

    total_rounds = sum(
        1 for d in history for r in d.get("rounds", [])
        if r.get("winner") not in ("tie", None)
    )
    status = {
        "validator_started_at": validator_started_at,
        "netuid": config.validate_netuid,
        "scoring": {
            "method": "race",
            "duel_rounds": config.validate_duel_rounds,
            "win_margin": config.validate_win_margin,
            "similarity_score_weight": 0.0,
            "llm_diff_judge_weight": diff_judge_weight,
            "llm_diff_judge_model": ",".join(resolve_diff_judge_models_fn(config)),
            "llm_diff_judge_models": list(resolve_diff_judge_models_fn(config)),
            "ties_count": False,
            "description": "Round score is 100% dual LLM diff judgment; challenger must win more decisive rounds than the king plus margin (ties ignored)",
        },
        "queue": [
            {
                "uid": s.uid,
                "repo": s.repo_full_name,
                "hotkey": s.hotkey,
                "commitment_block": s.commitment_block,
                "source": s.source,
                "pr_number": s.pr_number,
                "pr_url": s.pr_url,
            }
            for s in state.queue
        ],
        "active_duel": active_duel_info,
        "links": links,
        "disqualified": [_resolve_hk(hk) for hk in state.disqualified_hotkeys],
        "retired": [_resolve_hk(hk) for hk in state.retired_hotkeys],
        "total_rounds": total_rounds,
        "miners_seen": len(state.seen_hotkeys),
        "king_since": state.king_since,
        "king_duels_defended": state.king_duels_defended,
        "king_window_size": config.validate_king_window_size,
        "recent_kings": [
            dashboard_submission_dict_fn(
                k,
                history=history,
                share=1.0 / max(1, config.validate_king_window_size),
            )
            for k in effective_recent_kings_fn(state)
        ],
        "chain_data": chain_data,
    }

    payload = {
        "updated_at": timestamp_fn(),
        "current_king": king_dict,
        "duels": history,
        "status": status,
        "links": links,
    }
    try:
        write_json(config.validate_root / "dashboard_data.json", payload)
    except Exception:
        log.exception("Local dashboard write failed (non-fatal)")
    try:
        publish_dashboard_data(current_king=king_dict, duel_history=history, status=status)
    except Exception:
        log.exception("R2 dashboard publish failed (non-fatal)")


def _dashboard_links() -> dict[str, str]:
    public_base_url = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
    if public_base_url and not public_base_url.endswith("/sn66"):
        public_base_url = f"{public_base_url}/sn66"
    duels_html = f"{public_base_url}/duels.html" if public_base_url else "duels.html"
    return {"duels_html": duels_html}


def _dashboard_submission_dict(
    submission: Any,
    *,
    history: list[dict[str, Any]] | None = None,
    share: float | None = None,
    github_pr_merged_source: str,
) -> dict[str, Any]:
    display_repo = submission.repo_full_name
    display_commit = submission.commit_sha
    display_url = submission.pr_url or f"https://github.com/{display_repo}"
    winning_summary = _find_winning_challenger_summary(
        submission,
        history or [],
        github_pr_merged_source=github_pr_merged_source,
    )

    if winning_summary is not None:
        display_repo = str(winning_summary.get("challenger_repo") or display_repo)
        display_commit = str(winning_summary.get("challenger_commit_sha") or display_commit)
        display_url = str(
            winning_summary.get("challenger_pr_url")
            or submission.pr_url
            or winning_summary.get("challenger_repo_url")
            or f"https://github.com/{display_repo}"
        )

    payload = {
        "uid": submission.uid,
        "hotkey": submission.hotkey,
        "repo": display_repo,
        "repo_full_name": display_repo,
        "repo_url": display_url,
        "commit_sha": display_commit,
        "display_repo_full_name": display_repo,
        "display_repo_url": display_url,
        "display_commit_sha": display_commit,
        "runtime_repo_full_name": submission.repo_full_name,
        "runtime_repo_url": f"https://github.com/{submission.repo_full_name}",
        "runtime_commit_sha": submission.commit_sha,
        "source": submission.source,
        "pr_number": submission.pr_number,
        "pr_url": submission.pr_url,
    }
    if share is not None:
        payload["share"] = share
    return payload


def _find_winning_challenger_summary(
    submission: Any,
    history: list[dict[str, Any]],
    *,
    github_pr_merged_source: str,
) -> dict[str, Any] | None:
    if submission.source != github_pr_merged_source:
        return None
    for duel in reversed(history):
        if not duel.get("king_replaced"):
            continue
        if duel.get("challenger_hotkey") != submission.hotkey:
            continue
        try:
            if int(duel.get("challenger_uid")) != int(submission.uid):
                continue
        except (TypeError, ValueError):
            continue
        return duel
    return None
