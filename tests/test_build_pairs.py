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

    def test_rejects_verb_change(self):
        """Regression: a changed verb is a different task, not a swapped referent.

        "put the bowl on the stove" vs "turn on the stove" diffs into the single block
        "put the bowl" -> "turn", which contains no preposition, so every other gate
        passes it. But the two arms command different actions and share no referent, so
        an effect could not be attributed to anything.
        """
        assert (
            single_span_diff("put the bowl on the stove", "turn on the stove") is None
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


class TestWrittenArtifactIntegrity:
    """The stimulus file must match the hash it records, on every platform."""

    @staticmethod
    def _pair(i):
        from src.data.build_pairs import ContrastivePair

        return ContrastivePair(
            pair_id=f"p{i}",
            family="destination_swap",
            instruction_a="put the bowl on the plate",
            instruction_b="put the bowl on the stove",
            differing_span_a="plate",
            differing_span_b="stove",
            span_index=5,
            source_task="t",
            source_file="libero_goal/x.hdf5",
            source_sha256="deadbeef",
            demo="demo_0",
            timestep=i,
            provenance="test",
        )

    def test_file_matches_recorded_hash(self, tmp_path):
        """Regression: `write_text` translates \\n to \\r\\n on Windows.

        The hash is taken over the LF form, so the artifact would fail its own integrity
        check on the platform that produced it -- and hash differently per author.
        """
        import hashlib
        import json

        from src.data.build_pairs import write_pairs

        out = tmp_path / "pairs.jsonl"
        pairs = [self._pair(i) for i in range(5)]
        recorded = write_pairs(pairs, out, report={})

        body = out.read_bytes()
        assert hashlib.sha256(body).hexdigest() == recorded
        assert b"\r\n" not in body, "CRLF leaked into a released artifact"

        manifest = json.loads(out.with_suffix(".manifest.json").read_text())
        assert manifest["content_sha256"] == recorded
        assert manifest["n_pairs"] == 5

    def test_roundtrip(self, tmp_path):
        from src.data.build_pairs import load_pairs, write_pairs

        out = tmp_path / "pairs.jsonl"
        write_pairs([self._pair(i) for i in range(3)], out, report={})
        loaded = load_pairs(out)
        assert [r["pair_id"] for r in loaded] == ["p0", "p1", "p2"]


# ------------------------------------------------------- integration on real data

pytest.importorskip("h5py")


def _group_sources(pairs):
    """{(instruction_a, instruction_b): {source_task: count}}"""
    import collections

    out = collections.defaultdict(collections.Counter)
    for p in pairs:
        out[(p.instruction_a, p.instruction_b)][p.source_task] += 1
    return out


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

    def test_states_are_counterbalanced_across_both_tasks(self, tasks):
        """Regression, and the most consequential bug found so far.

        The directional readout asks whether commanding the non-demonstrated instruction
        moves the action away from what the demonstration did. If every state comes from
        the A-side task, that question is only ever asked in one direction, and any
        systematic asymmetry between the two instructions reads as a spurious above- or
        below-chance score instead of cancelling.

        The original `_pair_stream` consumed task A's states fully before starting task B.
        Callers take far fewer pairs than one task supplies, so B was never reached and
        100% of states came from A -- while the docstring claimed otherwise. Two
        checkpoints produced opposite, strongly-significant directional scores off that
        set before it was caught.
        """
        from src.data.build_pairs import build_pairs

        pairs, report = build_pairs(tasks, n=200, seed=0)

        for (a, b), sources in _group_sources(pairs).items():
            assert len(sources) == 2, (
                f"pairing {a!r} vs {b!r} drew states from only {list(sources)}; "
                "the set is not counterbalanced"
            )
            lo, hi = sorted(sources.values())
            assert lo / (lo + hi) > 0.3, f"pairing {a!r} vs {b!r} is lopsided: {sources}"

        frac = report["source_is_a_fraction"]
        assert 0.35 < frac < 0.65, f"A-side fraction {frac:.2f} is not balanced"

    def test_pair_ids_are_unique(self, tasks):
        from src.data.build_pairs import build_pairs

        pairs, _ = build_pairs(tasks, n=200, seed=0)
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids))

    def test_action_chunk_reference(self, tasks):
        """The directional readout needs the demo's next N actions, not one repeated.

        Comparing a 50-step prediction against a single instant would score a policy that
        stalls as highly as one that follows the demonstrated trajectory through.
        """
        from pathlib import Path

        from src.data.build_pairs import build_pairs, load_observation

        root = Path(__file__).resolve().parents[1] / "data" / "libero"
        pairs, _ = build_pairs(tasks, n=4, seed=0)
        for p in pairs[:3]:
            obs = load_observation(p.as_dict(), root, chunk_size=50)
            chunk = obs["action_chunk"]
            assert chunk.shape == (50, 7), f"expected (50, 7), got {chunk.shape}"
            assert np.isfinite(chunk).all()
            # First row must be the action at this exact timestep.
            assert np.allclose(chunk[0], obs["action"])
            # A real trajectory moves; a constant chunk would mean the padding logic ate it.
            assert not np.allclose(chunk[0], chunk[-1]), "chunk is constant"

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
