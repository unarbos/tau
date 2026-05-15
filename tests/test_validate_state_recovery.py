import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from validate import (
    ActiveDuelLease,
    PoolTask,
    RunConfig,
    TaskPool,
    ValidatePaths,
    ValidationRoundResult,
    ValidatorState,
    ValidatorSubmission,
    _checkpoint_active_duel,
    _active_duel_dashboard_info_from_state,
    _build_recent_kings_for_r2_publish,
    _publish_dashboard,
    _purge_stale_recent_kings_after_restart,
    _reconcile_dashboard_history_with_duels,
    _reconcile_state_with_duel_history,
    _recover_active_duel_after_restart,
    _pop_resumable_active_challenger,
    _replay_local_duel_files_to_r2,
    republish_recent_kings_dashboard_to_r2,
    _run_parallel_duel,
    _start_active_duel,
    _upsert_dashboard_history_summary,
)


class ValidatorStateRecoveryTest(unittest.TestCase):
    def test_startup_purge_clears_recent_kings_when_current_king_missing(self):
        previous = _submission(
            hotkey="5PreviousKing",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        state = ValidatorState(
            current_king=None,
            recent_kings=[previous],
            king_since="2026-05-11T14:37:39+00:00",
            king_duels_defended=7,
        )

        changed = _purge_stale_recent_kings_after_restart(state)

        self.assertTrue(changed)
        self.assertEqual(state.recent_kings, [])
        self.assertIsNone(state.king_since)
        self.assertEqual(state.king_duels_defended, 0)

    def test_build_recent_kings_for_r2_publish_reconstructs_from_duels(self):
        king_a = _submission(
            hotkey="5KingA",
            uid=1,
            commitment="unarbos/ninja@" + "a" * 40,
            block=101,
        )
        king_b = _submission(
            hotkey="5KingB",
            uid=2,
            commitment="unarbos/ninja@" + "b" * 40,
            block=102,
        )
        king_c = _submission(
            hotkey="5KingC",
            uid=3,
            commitment="unarbos/ninja@" + "c" * 40,
            block=103,
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "000002.json").write_text(
                json.dumps({"duel_id": 2, "king_before": king_a.to_dict(), "king_after": king_b.to_dict()}) + "\n"
            )
            (duels_dir / "000003.json").write_text(
                json.dumps({"duel_id": 3, "king_before": king_b.to_dict(), "king_after": king_c.to_dict()}) + "\n"
            )

            recent = _build_recent_kings_for_r2_publish(
                state=ValidatorState(),
                duels_dir=duels_dir,
                window=3,
            )

        self.assertEqual([submission.uid for submission in recent], [3, 2, 1])

    def test_republish_recent_kings_dashboard_to_r2_uses_reconstructed_window(self):
        king_a = _submission(
            hotkey="5KingA",
            uid=1,
            commitment="unarbos/ninja@" + "a" * 40,
            block=101,
        )
        king_b = _submission(
            hotkey="5KingB",
            uid=2,
            commitment="unarbos/ninja@" + "b" * 40,
            block=102,
        )
        king_c = _submission(
            hotkey="5KingC",
            uid=3,
            commitment="unarbos/ninja@" + "c" * 40,
            block=103,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validate_root = root / "workspace" / "validate" / "netuid-66"
            duels_dir = validate_root / "duels"
            duels_dir.mkdir(parents=True)
            (validate_root / "task-pool").mkdir()
            (validate_root / "task-pool-retest").mkdir()
            (validate_root / "state.json").write_text(json.dumps(ValidatorState().to_dict()) + "\n")
            (validate_root / "dashboard_history.json").write_text(json.dumps([]) + "\n")
            (duels_dir / "000002.json").write_text(
                json.dumps({"duel_id": 2, "king_before": king_a.to_dict(), "king_after": king_b.to_dict()}) + "\n"
            )
            (duels_dir / "000003.json").write_text(
                json.dumps({"duel_id": 3, "king_before": king_b.to_dict(), "king_after": king_c.to_dict()}) + "\n"
            )

            with mock.patch("validate.publish_dashboard_data", return_value=True):
                result = republish_recent_kings_dashboard_to_r2(
                    config=RunConfig(workspace_root=root, validate_netuid=66),
                    count=3,
                    set_current_from_history=True,
                )

            payload = json.loads((validate_root / "dashboard_data.json").read_text())

        self.assertEqual(result["recent_king_uids"], [3, 2, 1])
        self.assertEqual(result["current_king_uid"], 3)
        self.assertEqual([item["uid"] for item in payload["status"]["recent_kings"]], [3, 2, 1])
        self.assertEqual(payload["current_king"]["uid"], 3)

    def test_dashboard_recent_kings_include_reign_timestamps(self):
        first = _submission(
            hotkey="5FirstKing",
            uid=1,
            commitment="unarbos/ninja@" + "a" * 40,
            block=101,
        )
        previous = _submission(
            hotkey="5PreviousKing",
            uid=2,
            commitment="unarbos/ninja@" + "b" * 40,
            block=102,
        )
        current = _submission(
            hotkey="5CurrentKing",
            uid=3,
            commitment="unarbos/ninja@" + "c" * 40,
            block=103,
        )
        previous_transition = {
            "duel_id": 1,
            "started_at": "2026-05-10T18:00:00+00:00",
            "finished_at": "2026-05-10T18:10:00+00:00",
            "king_uid": first.uid,
            "king_hotkey": first.hotkey,
            "king_commit_sha": first.commit_sha,
            "challenger_uid": previous.uid,
            "challenger_hotkey": previous.hotkey,
            "challenger_commit_sha": previous.commit_sha,
            "king_replaced": True,
        }
        transition = {
            "duel_id": 3,
            "started_at": "2026-05-10T20:00:00+00:00",
            "finished_at": "2026-05-10T20:10:00+00:00",
            "king_before": previous.to_dict(),
            "challenger": current.to_dict(),
            "king_after": current.to_dict(),
            "king_replaced": True,
            "confirmation_duel_id": 4,
            "confirmation_retest_passed": True,
        }
        confirmation = {
            "duel_id": 4,
            "started_at": "2026-05-10T20:11:00+00:00",
            "finished_at": "2026-05-10T20:20:00+00:00",
            "king_before": previous.to_dict(),
            "challenger": current.to_dict(),
            "king_after": previous.to_dict(),
            "king_replaced": False,
            "confirmation_of_duel_id": 3,
            "task_set_phase": "confirmation_retest",
        }
        state = ValidatorState(
            current_king=current,
            recent_kings=[current, previous],
            king_since="2026-05-10T20:25:00+00:00",
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch("validate.publish_dashboard_data", return_value=True):
            config = RunConfig(workspace_root=Path(tmp), validate_king_window_size=2)
            config.validate_root.mkdir(parents=True)
            _publish_dashboard(
                state,
                [previous_transition, transition, confirmation],
                config,
                validator_started_at="2026-05-10T19:00:00+00:00",
            )
            payload = json.loads((config.validate_root / "dashboard_data.json").read_text())

        recent = payload["status"]["recent_kings"]
        self.assertEqual(recent[0]["uid"], 3)
        self.assertEqual(recent[0]["king_since"], "2026-05-10T20:25:00+00:00")
        self.assertEqual(recent[1]["uid"], 2)
        self.assertEqual(recent[1]["king_since"], "2026-05-10T18:10:00+00:00")

    def test_dashboard_uses_private_submission_repo_label(self):
        private = ValidatorSubmission(
            hotkey="5PrivateHotkey",
            uid=7,
            repo_full_name="private-submission/5PrivateHotkey-deadbeef",
            repo_url="private-submission://5PrivateHotkey-deadbeef",
            commit_sha="d" * 64,
            commitment="private-submission:5PrivateHotkey-deadbeef:" + "d" * 64,
            commitment_block=123,
            source="private",
        )
        state = ValidatorState(
            current_king=private,
            queue=[private],
            active_duel=ActiveDuelLease(
                duel_id=10,
                started_at="2026-05-10T18:00:00+00:00",
                king=private,
                challenger=private,
                task_names=["validate-000001"],
            ),
            retired_hotkeys=[private.hotkey],
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch("validate.publish_dashboard_data", return_value=True):
            config = RunConfig(workspace_root=Path(tmp), validate_netuid=66)
            _publish_dashboard(
                state,
                [],
                config,
                "2026-05-10T18:00:00+00:00",
            )
            payload = json.loads((config.validate_root / "dashboard_data.json").read_text())

        self.assertEqual(payload["current_king"]["repo_full_name"], "private-submission")
        self.assertIsNone(payload["current_king"]["repo_url"])
        self.assertEqual(payload["current_king"]["runtime_repo_full_name"], "private-submission")
        self.assertIsNone(payload["current_king"]["runtime_repo_url"])
        self.assertEqual(payload["status"]["queue"][0]["repo"], private.hotkey)
        self.assertEqual(payload["status"]["active_duel"]["king_repo"], private.hotkey)
        self.assertIsNone(payload["status"]["active_duel"]["king_repo_url"])
        self.assertEqual(payload["status"]["active_duel"]["challenger_repo"], private.hotkey)
        self.assertIsNone(payload["status"]["active_duel"]["challenger_repo_url"])
        self.assertEqual(payload["status"]["retired"][0]["repo"], "private-submission")

    def test_dashboard_recomputes_current_king_defenses_from_history_hotkey(self):
        current = _submission(
            hotkey="5CurrentKing",
            uid=3,
            commitment="unarbos/ninja@" + "c" * 40,
            block=103,
        )
        other = _submission(
            hotkey="5OtherKing",
            uid=4,
            commitment="unarbos/ninja@" + "d" * 40,
            block=104,
        )
        history = [
            {
                "duel_id": 1,
                "king_hotkey": other.hotkey,
                "challenger_hotkey": current.hotkey,
                "king_replaced": True,
            },
            {
                "duel_id": 2,
                "king_hotkey": current.hotkey,
                "challenger_hotkey": "5LoserA",
                "king_replaced": False,
            },
            {
                "duel_id": 3,
                "king_hotkey": current.hotkey,
                "challenger_hotkey": "5Retest",
                "king_replaced": False,
                "task_set_phase": "confirmation_retest",
                "confirmation_of_duel_id": 2,
            },
            {
                "duel_id": 4,
                "king_hotkey": other.hotkey,
                "challenger_hotkey": "5LoserB",
                "king_replaced": False,
            },
            {
                "duel_id": 5,
                "king_hotkey": current.hotkey,
                "challenger_hotkey": "5LoserC",
                "king_replaced": False,
            },
        ]
        state = ValidatorState(current_king=current, king_duels_defended=99)

        with tempfile.TemporaryDirectory() as tmp, mock.patch("validate.publish_dashboard_data", return_value=True):
            config = RunConfig(workspace_root=Path(tmp), validate_netuid=66)
            _publish_dashboard(
                state,
                history,
                config,
                "2026-05-10T18:00:00+00:00",
            )
            payload = json.loads((config.validate_root / "dashboard_data.json").read_text())

        self.assertEqual(payload["status"]["king_duels_defended"], 2)

    def test_dashboard_private_published_king_repo_label_stays_private_submission(self):
        king = ValidatorSubmission(
            hotkey="5PrivatePublished",
            uid=163,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha="5" * 40,
            commitment="private-submission:5PrivatePublished-deadbeef:" + "d" * 64,
            commitment_block=123,
            source="private_published",
            display_repo_full_name="unarbos/ninja",
            display_commit_sha="5" * 40,
        )
        state = ValidatorState(current_king=king, recent_kings=[king])

        with tempfile.TemporaryDirectory() as tmp, mock.patch("validate.publish_dashboard_data", return_value=True):
            config = RunConfig(workspace_root=Path(tmp), validate_netuid=66)
            _publish_dashboard(
                state,
                [],
                config,
                "2026-05-10T18:00:00+00:00",
            )
            payload = json.loads((config.validate_root / "dashboard_data.json").read_text())

        self.assertEqual(payload["current_king"]["repo_full_name"], "private-submission")
        self.assertEqual(payload["current_king"]["runtime_repo_full_name"], "unarbos/ninja")
        self.assertEqual(payload["status"]["recent_kings"][0]["repo_full_name"], "private-submission")
        self.assertEqual(payload["status"]["recent_kings"][0]["runtime_repo_full_name"], "unarbos/ninja")

    def test_reconcile_advances_duel_id_and_removes_completed_queue_entry(self):
        completed = _submission(
            hotkey="5CompletedHotkey",
            uid=210,
            commitment="unarbos/ninja@" + "a" * 40,
            block=123,
        )
        pending = _submission(
            hotkey="5PendingHotkey",
            uid=97,
            commitment="unarbos/ninja@" + "b" * 40,
            block=124,
        )
        state = ValidatorState(
            queue=[completed, pending],
            next_duel_index=3990,
            seen_hotkeys=[],
            locked_commitments={},
            commitment_blocks_by_hotkey={},
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "003990.json").write_text(
                json.dumps(
                    {
                        "duel_id": 3990,
                        "challenger": completed.to_dict(),
                    }
                )
                + "\n"
            )

            changed = _reconcile_state_with_duel_history(state, duels_dir)

        self.assertTrue(changed)
        self.assertEqual(state.next_duel_index, 3991)
        self.assertEqual([s.hotkey for s in state.queue], [pending.hotkey])
        self.assertIn(completed.hotkey, state.seen_hotkeys)
        self.assertEqual(state.locked_commitments[completed.hotkey], completed.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[completed.hotkey], completed.commitment_block)

    def test_reconcile_dashboard_history_appends_missing_local_duels(self):
        existing = {"duel_id": 3989, "wins": 1, "losses": 0}
        challenger = _submission(
            hotkey="5CompletedHotkey",
            uid=210,
            commitment="unarbos/ninja@" + "a" * 40,
            block=123,
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "003990.json").write_text(
                json.dumps(
                    {
                        "duel_id": 3990,
                        "started_at": "2026-05-05T00:00:00+00:00",
                        "finished_at": "2026-05-05T00:01:00+00:00",
                        "king_before": challenger.to_dict(),
                        "challenger": challenger.to_dict(),
                        "king_after": challenger.to_dict(),
                        "rounds": [],
                        "wins": 0,
                        "losses": 5,
                        "ties": 0,
                        "king_replaced": False,
                    }
                )
                + "\n"
            )
            history = [existing]

            changed = _reconcile_dashboard_history_with_duels(history, duels_dir)

        self.assertTrue(changed)
        self.assertEqual([entry["duel_id"] for entry in history], [3989, 3990])
        self.assertEqual(history[0], existing)
        self.assertEqual(history[1]["challenger_hotkey"], challenger.hotkey)
        self.assertEqual(history[1]["losses"], 5)

    def test_upsert_dashboard_history_summary_replaces_same_duel(self):
        history = [
            {"duel_id": 4221, "wins": 28, "losses": 21, "task_set_phase": "primary"},
            {"duel_id": 4222, "wins": 27, "losses": 21, "task_set_phase": "confirmation_retest"},
        ]

        changed = _upsert_dashboard_history_summary(
            history,
            {
                "duel_id": 4221,
                "wins": 28,
                "losses": 21,
                "task_set_phase": "primary",
                "confirmation_duel_id": 4222,
                "confirmation_retest_passed": True,
            },
        )

        self.assertFalse(changed)
        self.assertEqual([entry["duel_id"] for entry in history], [4221, 4222])
        self.assertEqual(history[0]["confirmation_duel_id"], 4222)
        self.assertTrue(history[0]["confirmation_retest_passed"])

    def test_r2_replay_publishes_local_duel_files_and_index_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duels_dir = root / "duels"
            duels_dir.mkdir()
            (duels_dir / "000001.json").write_text(json.dumps({"duel_id": 1}) + "\n")
            (duels_dir / "000002.json").write_text(json.dumps({"duel_id": 2}) + "\n")

            with (
                mock.patch("validate.publish_duel_data", return_value=True) as publish_duel,
                mock.patch("validate.publish_duel_index", return_value=True) as publish_index,
            ):
                _replay_local_duel_files_to_r2(
                    _paths_for_duels(root, duels_dir),
                    [{"duel_id": 1}, {"duel_id": 2}],
                )

        self.assertEqual(
            [call.kwargs["duel_id"] for call in publish_duel.call_args_list],
            [2, 1],
        )
        self.assertEqual(publish_index.call_args.kwargs["latest_duel_dict"]["duel_id"], 2)

    def test_active_duel_lease_round_trips_through_state(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        round_result = _round(task_name="validate-000001", winner="challenger")
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=77,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[round_result],
                status="running",
                updated_at="2026-05-06T00:01:00+00:00",
            ),
        )

        restored = ValidatorState.from_dict(state.to_dict())

        self.assertIsNotNone(restored.active_duel)
        assert restored.active_duel is not None
        self.assertEqual(restored.active_duel.duel_id, 77)
        self.assertEqual(restored.active_duel.challenger.hotkey, challenger.hotkey)
        self.assertEqual(restored.active_duel.task_names, ["validate-000001", "validate-000002"])
        self.assertEqual(restored.active_duel.rounds[0].winner, "challenger")
        self.assertIn(challenger.hotkey, restored.seen_hotkeys)

    def test_checkpoint_active_duel_updates_tasks_rounds_and_status(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState()
        _start_active_duel(state, duel_id=81, king=king, challenger=challenger)
        round_result = _round(task_name="validate-000003", winner="king")

        changed = _checkpoint_active_duel(
            state,
            duel_id=81,
            task_names=["validate-000003"],
            rounds=[round_result],
            status="draining",
        )

        self.assertTrue(changed)
        self.assertIsNotNone(state.active_duel)
        assert state.active_duel is not None
        self.assertEqual(state.active_duel.task_names, ["validate-000003"])
        self.assertEqual(state.active_duel.rounds, [round_result])
        self.assertEqual(state.active_duel.status, "draining")
        self.assertFalse(
            _checkpoint_active_duel(
                state,
                duel_id=82,
                task_names=["validate-000004"],
            )
        )

    def test_recover_active_duel_requeues_interrupted_challenger_at_front(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        pending = _submission(
            hotkey="5PendingHotkey",
            uid=13,
            commitment="unarbos/ninja@" + "c" * 40,
            block=113,
        )
        state = ValidatorState(
            current_king=king,
            queue=[pending],
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                rounds=[_round(task_name="validate-000001", winner="challenger")],
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            changed = _recover_active_duel_after_restart(
                config=RunConfig(),
                state=state,
                duels_dir=Path(tmp),
            )

        self.assertTrue(changed)
        self.assertIsNone(state.active_duel)
        self.assertEqual([s.hotkey for s in state.queue], [challenger.hotkey, pending.hotkey])
        self.assertEqual(state.current_king, king)
        self.assertNotIn(challenger.hotkey, state.disqualified_hotkeys)

    def test_recover_active_duel_preserves_resumable_primary_checkpoint(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            next_duel_index=91,
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[_round(task_name="validate-000001", winner="challenger")],
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            changed = _recover_active_duel_after_restart(
                config=RunConfig(),
                state=state,
                duels_dir=Path(tmp),
            )

        self.assertTrue(changed)
        self.assertIsNotNone(state.active_duel)
        assert state.active_duel is not None
        self.assertEqual(state.active_duel.duel_id, 90)
        self.assertEqual(state.next_duel_index, 90)
        self.assertEqual([s.hotkey for s in state.queue], [challenger.hotkey])

    def test_recover_active_duel_preserves_selected_tasks_before_any_round_scores(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            next_duel_index=91,
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[],
                status="tasks_selected",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            changed = _recover_active_duel_after_restart(
                config=RunConfig(),
                state=state,
                duels_dir=Path(tmp),
            )

        self.assertTrue(changed)
        self.assertIsNotNone(state.active_duel)
        assert state.active_duel is not None
        self.assertEqual(state.active_duel.duel_id, 90)
        self.assertEqual(state.next_duel_index, 90)
        self.assertEqual(state.active_duel.task_names, ["validate-000001", "validate-000002"])

    def test_resumable_active_duel_consumes_original_duel_id_when_index_drifted(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            next_duel_index=91,
            queue=[challenger],
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[],
                status="running_rounds",
            ),
        )

        resumed = _pop_resumable_active_challenger(state, king=king)

        self.assertIsNotNone(resumed)
        assert resumed is not None
        duel_id, resumed_challenger = resumed
        self.assertEqual(duel_id, 90)
        self.assertEqual(resumed_challenger.hotkey, challenger.hotkey)
        self.assertEqual(state.next_duel_index, 91)
        self.assertEqual(state.queue, [])

    def test_dashboard_reconstructs_active_duel_from_state_after_restart(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000001", "validate-000002"],
                rounds=[],
                status="running_rounds",
            ),
        )

        active = _active_duel_dashboard_info_from_state(
            state,
            history=[],
            config=RunConfig(validate_duel_rounds=50, validate_win_margin=3),
        )

        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active["duel_id"], 90)
        self.assertEqual(active["challenger_uid"], 12)
        self.assertEqual(active["phase"], "running_rounds")
        self.assertEqual(active["scored"], 0)
        self.assertEqual(active["gathered_tasks"], 2)
        self.assertEqual(active["needed_tasks"], 50)

    def test_parallel_duel_reuses_selected_tasks_with_zero_prior_scores(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=90,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
                task_names=["validate-000002", "validate-000003"],
                rounds=[],
                status="tasks_selected",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = TaskPool(root / "pool")
            for name, elapsed in (
                ("validate-000001", 1.0),
                ("validate-000002", 2.0),
                ("validate-000003", 3.0),
            ):
                pool.add(_pool_task(name, elapsed=elapsed))
            config = RunConfig(
                workspace_root=root,
                validate_duel_rounds=2,
                validate_round_concurrency=1,
                validate_win_margin=0,
            )

            with mock.patch("validate._solve_and_compare_round") as solve_round:
                solve_round.side_effect = lambda *, task, **_: _round(
                    task_name=task.task_name,
                    winner="challenger",
                )
                result = _run_parallel_duel(
                    config=config,
                    state=state,
                    king=king,
                    challenger=challenger,
                    duel_id=90,
                    pool=pool,
                )

        self.assertEqual(
            [call.kwargs["task"].task_name for call in solve_round.call_args_list],
            ["validate-000002", "validate-000003"],
        )
        self.assertEqual(
            [round_result.task_name for round_result in result.rounds],
            ["validate-000002", "validate-000003"],
        )

    def test_recover_active_duel_clears_lease_when_duel_file_exists(self):
        king = _submission(
            hotkey="5KingHotkey",
            uid=11,
            commitment="unarbos/ninja@" + "a" * 40,
            block=111,
        )
        challenger = _submission(
            hotkey="5ChallengerHotkey",
            uid=12,
            commitment="unarbos/ninja@" + "b" * 40,
            block=112,
        )
        state = ValidatorState(
            current_king=king,
            active_duel=ActiveDuelLease(
                duel_id=91,
                started_at="2026-05-06T00:00:00+00:00",
                king=king,
                challenger=challenger,
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "000091.json").write_text(json.dumps({"duel_id": 91}) + "\n")
            changed = _recover_active_duel_after_restart(
                config=RunConfig(),
                state=state,
                duels_dir=duels_dir,
            )

        self.assertTrue(changed)
        self.assertIsNone(state.active_duel)
        self.assertEqual(state.queue, [])


def _submission(*, hotkey: str, uid: int, commitment: str, block: int) -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=hotkey,
        uid=uid,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=commitment.rsplit("@", 1)[-1],
        commitment=commitment,
        commitment_block=block,
        source="chain",
        base_repo_full_name="unarbos/ninja",
        base_ref="main",
    )


def _round(*, task_name: str, winner: str) -> ValidationRoundResult:
    return ValidationRoundResult(
        task_name=task_name,
        winner=winner,
        king_lines=10,
        challenger_lines=12,
        king_similarity_ratio=0.5,
        challenger_similarity_ratio=0.7,
        king_challenger_similarity=0.4,
        task_root=f"/tmp/{task_name}",
        king_compare_root="",
        challenger_compare_root="",
    )


def _pool_task(name: str, *, elapsed: float) -> PoolTask:
    return PoolTask(
        task_name=name,
        task_root=f"/tmp/{name}",
        creation_block=1,
        cursor_elapsed=elapsed,
        king_lines=10,
        king_similarity=0.5,
        baseline_lines=10,
    )


def _paths_for_duels(root: Path, duels_dir: Path) -> ValidatePaths:
    return ValidatePaths(
        root=root,
        state_path=root / "state.json",
        duels_dir=duels_dir,
        pool_dir=root / "task-pool",
        retest_pool_dir=root / "task-pool-retest",
    )


if __name__ == "__main__":
    unittest.main()
