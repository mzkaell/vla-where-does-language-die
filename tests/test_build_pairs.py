"""Tests for contrastive pair construction (M1).

The stimulus set is a released artifact and every causal claim rests on it. If pairs are
not truly minimal -- if two things differ instead of one -- then any measured effect is
unattributable and the localization map is meaningless. These tests target that.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.build_pairs import (
    GRIPPER_ACTION_DIM,
    classify_family,
    pre_grasp_timesteps,
    single_span_diff,
)


class TestSingleSpanDiff:
    def test_detects_single_word_swap(self):
        got = single_span_diff(
            "put the bowl on the plate",
            "put the bowl on the stove",
        )
        assert got == (5, "plate", "stove")

    def test_detects_multiword_referent_swap(self):
        """A multi-word referent is still ONE referent."""
        got = single_span_diff(
            "put the wine bottle on the rack",
            "put the wine bottle on top of the cabinet",
        )
        assert got is not None
        idx, a, b = got
        assert a == "the rack"
        assert b == "top of the cabinet"

    def test_detects_object_swap(self):
        got = single_span_diff(
            "put the bowl on top of the cabinet",
            "put the wine bottle on top of the cabinet",
        )
        assert got is not None
        _, a, b = got
        assert a == "bowl"
        assert b == "wine bottle"

    def test_rejects_two_separate_edits(self):
        """The core validity gate.

        Two edits mean an observed effect cannot be attributed to either swap, so such a
        pairing must never enter the stimulus set.
        """
        got = single_span_diff(
            "put the bowl on the plate",
            "put the mug on the stove",
        )
        assert got is None

    def test_rejects_object_and_destination_both_changing(self):
        """Regression: the filler-merge rule used to let this through.

        Both the object (cream cheese -> wine bottle) and the destination (bowl -> rack)
        change, but the two edit blocks are separated only by the shared determiner
        "the", so merging across fillers fused them into one apparent span. A pair like
        this is unattributable and must never ship.
        """
        assert (
            single_span_diff(
                "put the cream cheese in the bowl",
                "put the wine bottle on the rack",
            )
            is None
        )

    def test_rejects_object_and_destination_both_changing_variant(self):
        assert (
            single_span_diff(
                "put the cream cheese in the bowl",
                "put the wine bottle on top of the cabinet",
            )
            is None
        )

    def test_rejects_pure_prefix(self):
        """"turn on the stove" vs "turn on the stove now" is an addition, not a swap."""
        assert single_span_diff("turn on the stove", "turn on the stove now") is None

    def test_identical_returns_none(self):
        assert single_span_diff("put the bowl on the plate", "put the bowl on the plate") is None

    def test_is_symmetric_in_detection(self):
        a = "put the bowl on the plate"
        b = "put the bowl on the stove"
        assert (single_span_diff(a, b) is None) == (single_span_diff(b, a) is None)


class TestFamilyClassification:
    def test_destination_swap(self):
        assert (
            classify_family("plate", "stove", "put the bowl on the plate") == "destination_swap"
        )

    def test_object_swap(self):
        assert (
            classify_family("bowl", "wine bottle", "put the bowl on top of the cabinet")
            == "object_swap"
        )


class TestPreGraspTimesteps:
    @staticmethod
    def _demo(actions):
        return {"actions": np.asarray(actions, dtype=np.float64)}

    def test_stops_at_first_gripper_close(self):
        """States after the grasp are NOT valid pairs: the trajectory has committed."""
        actions = np.zeros((20, 7))
        actions[:, GRIPPER_ACTION_DIM] = -1.0
        actions[10:, GRIPPER_ACTION_DIM] = 1.0  # close at t=10

        ts = pre_grasp_timesteps(self._demo(actions), max_per_demo=100, stride=1)
        assert ts, "expected some pre-grasp states"
        assert max(ts) < 10, f"leaked a post-grasp timestep: {max(ts)}"

    def test_respects_stride_and_cap(self):
        actions = np.zeros((100, 7))
        actions[:, GRIPPER_ACTION_DIM] = -1.0
        ts = pre_grasp_timesteps(self._demo(actions), max_per_demo=3, stride=5)
        assert ts == [0, 5, 10]

    def test_no_close_uses_full_episode(self):
        actions = np.zeros((8, 7))
        actions[:, GRIPPER_ACTION_DIM] = -1.0
        ts = pre_grasp_timesteps(self._demo(actions), max_per_demo=100, stride=1)
        assert ts == list(range(8))

    def test_immediate_close_yields_nothing(self):
        actions = np.zeros((8, 7))
        actions[:, GRIPPER_ACTION_DIM] = 1.0
        assert pre_grasp_timesteps(self._demo(actions), max_per_demo=100, stride=1) == []


# ------------------------------------------------------- integration on real data

pytest.importorskip("h5py")


@pytest.mark.slow
class TestAgainstRealData:
    """Runs only when the LIBERO HDF5s are present."""

    @pytest.fixture(scope="class")
    def tasks(self):
        from pathlib import Path

        from src.data.build_pairs import discover_tasks

        root = Path(__file__).resolve().parents[1] / "data" / "libero" / "libero_goal"
        if not root.exists() or not list(root.glob("*.hdf5")):
            pytest.skip("LIBERO-Goal HDF5s not downloaded")
        return discover_tasks(root, compute_hash=False)

    def test_every_emitted_pair_is_minimal(self, tasks):
        """End-to-end validity: no pair in the output may differ in two places."""
        from src.data.build_pairs import build_pairs

        pairs, _ = build_pairs(tasks, n=120, seed=0)
        assert pairs
        for p in pairs:
            assert single_span_diff(p.instruction_a, p.instruction_b) is not None, (
                f"non-minimal pair emitted: {p.instruction_a!r} vs {p.instruction_b!r}"
            )
            assert p.instruction_a != p.instruction_b

    def test_pair_ids_are_unique(self, tasks):
        from src.data.build_pairs import build_pairs

        pairs, _ = build_pairs(tasks, n=200, seed=0)
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids))

    def test_observations_are_loadable_and_identical_across_arms(self, tasks):
        """The two arms of a pair must share pixels exactly -- that is the whole design."""
        from pathlib import Path

        from src.data.build_pairs import build_pairs, load_observation

        root = Path(__file__).resolve().parents[1] / "data" / "libero"
        pairs, _ = build_pairs(tasks, n=5, seed=0)
        for p in pairs[:3]:
            obs = load_observation(p.as_dict(), root)
            assert obs["agentview_rgb"].shape == (128, 128, 3)
            assert np.isfinite(obs["joint_states"]).all()
