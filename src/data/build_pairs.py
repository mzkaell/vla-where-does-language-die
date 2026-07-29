"""Minimal contrastive pairs on LIBERO-Goal (M1).

A pair is one visual state plus two instructions that differ in exactly one referent,
where **both instructions are individually achievable from that state**. Running the
policy on the same pixels under instruction A and instruction B isolates the causal
contribution of the swapped word, because literally nothing else differs.

Why LIBERO-Goal: its ten tasks share one kitchen scene with a fixed object set, and only
the commanded goal changes. So vision underdetermines the target and the instruction has
to be read. In LIBERO-Object or -Spatial the scene itself changes with the task, which
would confound the swap with a visual difference.

Why this is built offline from the released demonstration HDF5s
--------------------------------------------------------------
The demos already contain rendered observations (`agentview_rgb`, `eye_in_hand_rgb`) plus
proprioception, so pair construction and all downstream patching need no simulator. That
matters practically -- robosuite/MuJoCo headless is Linux-only -- but it is also better
science: fixed stored states are exactly reproducible, whereas re-rolling a simulator
introduces variance that has nothing to do with the instruction.

The stimulus file stores *references* (file, demo, timestep) rather than pixels. It stays
small and releasable, and the source HDF5s are public and content-hashed for integrity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = "libero-goal-pairs/1"

# LIBERO's gripper action channel: > 0 commands a close. Before the first close the
# robot has grasped nothing, so every task in the scene is still achievable from that
# state. This is the operational definition of "both actions individually valid".
GRIPPER_ACTION_DIM = 6


@dataclass(frozen=True)
class TaskSource:
    """One LIBERO task file and its instruction."""

    name: str
    path: Path
    instruction: str
    num_demos: int
    sha256: str


@dataclass
class ContrastivePair:
    """One released stimulus item."""

    pair_id: str
    family: str
    instruction_a: str
    instruction_b: str
    differing_span_a: str
    differing_span_b: str
    span_index: int
    source_task: str
    source_file: str
    source_sha256: str
    demo: str
    timestep: int
    provenance: str
    validation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- instructions


def single_span_diff(a: str, b: str) -> tuple[int, str, str] | None:
    """Return (word_index, span_a, span_b) iff `a` and `b` differ by exactly ONE edit.

    This is the formal version of "differs in exactly one referent", and it is the gate
    the whole design rests on: if two things differ, an observed action change cannot be
    attributed to either, and the localization map built on top means nothing.

    A naive "trim the common prefix and suffix" implementation is NOT sufficient, and
    that subtlety is worth stating because it looks sufficient. Consider

        put the bowl on the plate
        put the mug  on the stove

    The common prefix is "put the" and there is no common suffix, so trimming reports the
    single span "bowl on the plate" -> "mug on the stove". But that is two independent
    swaps (bowl->mug and plate->stove) that happen to share the words "on the" between
    them. A real word-level diff sees two replace blocks and rejects it.

    Multi-word referents ("the rack" -> "top of the cabinet") are still one edit and are
    accepted. Getting that case right needs one more step, because a raw word diff
    matches the shared determiner in the middle:

        the rack          ->  top of | the | cabinet
                                     ^^^^^^^

    and reports two blocks around it. So edit blocks separated *only* by filler words
    (determiners and "of") are merged back into one. Blocks separated by anything
    contentful -- notably the preposition in the bowl/mug example above -- are left
    separate and the pair is rejected.

    Pure insertions or deletions are rejected too: they lengthen or shorten the
    instruction rather than swapping a referent, so the arms would differ in more than
    the referent identity.
    """
    from difflib import SequenceMatcher

    wa, wb = a.split(), b.split()
    if wa == wb:
        return None

    merged: list[list[int]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if merged:
            prev = merged[-1]
            gap_a, gap_b = wa[prev[1] : i1], wb[prev[3] : j1]
            if all(w in _FILLERS for w in gap_a) and all(w in _FILLERS for w in gap_b):
                prev[1], prev[3] = i2, j2
                continue
        merged.append([i1, i2, j1, j2])

    if len(merged) != 1:
        return None  # zero, or several independent edits

    i1, i2, j1, j2 = merged[0]
    span_a, span_b = wa[i1:i2], wb[j1:j2]
    if not span_a or not span_b:
        return None  # a pure insertion or deletion

    # A referent is a noun phrase, so it must not span a preposition. Without this the
    # filler-merge above is too permissive in one specific way:
    #
    #     put the cream cheese in  the bowl
    #     put the wine bottle  on  the rack
    #
    # the object AND the destination both change, but the two edit blocks are separated
    # by the shared determiner "the", so they merge into one apparent span. Requiring the
    # span to be preposition-free rejects it, while still accepting a genuine multi-word
    # destination like "the rack" -> "top of the cabinet" ("of" is a filler, not a
    # preposition).
    if any(w in _PREPOSITIONS for w in span_a + span_b):
        return None

    # The edit must not touch the verb. LIBERO instructions open with the action
    # ("put", "open", "turn", "push"), so an edit starting at word 0 changes the task
    # rather than swapping a referent within it:
    #
    #     put the bowl on the stove   ->   turn on the stove
    #
    # That reads as one clean block ("put the bowl" -> "turn") with no preposition, so
    # every earlier gate passes it, yet the two arms command different actions entirely
    # and share no referent to attribute an effect to.
    if i1 == 0:
        return None
    return i1, " ".join(span_a), " ".join(span_b)


# Words that may sit between two edit blocks without making them separate referents.
# Deliberately narrow: determiners and "of" only. Adding prepositions here would let the
# two-swap case ("bowl on the plate" -> "mug on the stove") through as a single edit.
_FILLERS = {"the", "a", "an", "of"}

_PREPOSITIONS = {"on", "in", "to", "into", "inside", "onto", "at", "under", "behind"}


def classify_family(span_a: str, span_b: str, instruction_a: str, span_index: int = -1) -> str:
    """Label the swap: the manipulated object, or its destination.

    Decided by whether the edit falls before or after the first preposition, which is
    where LIBERO instructions switch from naming the object to naming the goal.
    """
    words = instruction_a.split()
    first_prep = next((i for i, w in enumerate(words) if w in _PREPOSITIONS), None)
    if first_prep is None:
        return "other_swap"

    if span_index < 0:
        first_word = span_a.split()[0]
        span_index = words.index(first_word) if first_word in words else 0
    return "object_swap" if span_index < first_prep else "destination_swap"


# ------------------------------------------------------------------------- loading


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_task(path: Path, compute_hash: bool = True) -> TaskSource:
    """Read one LIBERO demo file's metadata."""
    with h5py.File(path, "r") as f:
        data = f["data"]
        info = json.loads(data.attrs["problem_info"])
        instruction = info["language_instruction"].strip()
        num_demos = int(data.attrs["num_demos"])
    return TaskSource(
        name=path.stem.replace("_demo", ""),
        path=path,
        instruction=instruction,
        num_demos=num_demos,
        sha256=sha256_file(path) if compute_hash else "",
    )


def discover_tasks(root: Path, compute_hash: bool = True) -> list[TaskSource]:
    files = sorted(root.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"no LIBERO .hdf5 files under {root}. Download them first, e.g.\n"
            "  huggingface-cli download yifengzhu-hf/LIBERO-datasets --repo-type dataset "
            "--include 'libero_goal/*' --local-dir data/libero"
        )
    return [load_task(p, compute_hash=compute_hash) for p in files]


def pre_grasp_timesteps(demo: h5py.Group, max_per_demo: int, stride: int) -> list[int]:
    """Timesteps before the first gripper-close command.

    At these states nothing has been grasped yet, so every task in the shared scene is
    still achievable -- which is precisely the "both actions individually valid"
    requirement. After the grasp, the trajectory has committed and the alternative
    instruction is no longer reachable without regrasping.
    """
    actions = demo["actions"][:]
    closes = np.nonzero(actions[:, GRIPPER_ACTION_DIM] > 0)[0]
    horizon = int(closes[0]) if closes.size else actions.shape[0]
    if horizon <= 0:
        return []
    candidates = list(range(0, horizon, stride))
    return candidates[:max_per_demo]


# --------------------------------------------------------------------- construction


def build_pairs(
    tasks: Sequence[TaskSource],
    n: int,
    max_per_demo: int = 2,
    stride: int = 4,
    seed: int = 0,
) -> tuple[list[ContrastivePair], dict[str, Any]]:
    """Generate validated contrastive pairs, balanced across instruction pairings.

    Returns the pairs plus a report of what was accepted and what was rejected and why.
    Rejections are reported, not silently dropped -- a stimulus set whose failures are
    invisible cannot be audited.
    """
    rng = np.random.default_rng(seed)

    # Which instruction pairings are minimal (exactly one referent differs)?
    pairings: list[tuple[TaskSource, TaskSource, tuple[int, str, str], str]] = []
    rejected_pairings: list[dict[str, str]] = []
    for i, ta in enumerate(tasks):
        for tb in tasks[i + 1 :]:
            diff = single_span_diff(ta.instruction, tb.instruction)
            if diff is None:
                rejected_pairings.append(
                    {
                        "a": ta.instruction,
                        "b": tb.instruction,
                        "reason": "not a single contiguous referent swap",
                    }
                )
                continue
            family = classify_family(diff[1], diff[2], ta.instruction, span_index=diff[0])
            pairings.append((ta, tb, diff, family))

    if not pairings:
        raise RuntimeError("no minimal instruction pairs found among the supplied tasks")

    # Round-robin over pairings so no single swap dominates the stimulus set.
    pools: list[Iterator[ContrastivePair]] = [
        _pair_stream(ta, tb, diff, family, max_per_demo, stride, rng)
        for ta, tb, diff, family in pairings
    ]

    pairs: list[ContrastivePair] = []
    exhausted = set()
    while len(pairs) < n and len(exhausted) < len(pools):
        for k, stream in enumerate(pools):
            if k in exhausted or len(pairs) >= n:
                continue
            try:
                pairs.append(next(stream))
            except StopIteration:
                exhausted.add(k)

    report = {
        "requested": n,
        "produced": len(pairs),
        "instruction_pairings": [
            {
                "a": ta.instruction,
                "b": tb.instruction,
                "family": family,
                "swap": f"{diff[1]!r} -> {diff[2]!r}",
            }
            for ta, tb, diff, family in pairings
        ],
        "rejected_pairings": rejected_pairings,
        "counts_by_family": _counts(p.family for p in pairs),
        "counts_by_swap": _counts(f"{p.differing_span_a} -> {p.differing_span_b}" for p in pairs),
        # Counterbalancing: what fraction of states came from the A-side task. Must be
        # near 0.5. A one-sided set silently turns the directional readout into a test of
        # instruction asymmetry, which reads as a spurious above/below-chance score.
        "source_is_a_fraction": _source_balance(pairs),
        "counts_by_source_task": _counts(p.source_task for p in pairs),
    }
    return pairs, report


def _source_balance(pairs: Sequence[ContrastivePair]) -> float:
    if not pairs:
        return float("nan")
    a_side = sum(1 for p in pairs if p.source_task == p.pair_id.split("__")[0])
    return a_side / len(pairs)


def _counts(values: Iterator[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _states_from(
    source: TaskSource,
    max_per_demo: int,
    stride: int,
    rng: np.random.Generator,
) -> Iterator[tuple[TaskSource, str, int]]:
    """Yield (source, demo_name, timestep) for every usable pre-grasp state in a task."""
    with h5py.File(source.path, "r") as f:
        data = f["data"]
        demos = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
        for di in rng.permutation(len(demos)):
            demo_name = demos[int(di)]
            for t in pre_grasp_timesteps(data[demo_name], max_per_demo, stride):
                yield source, demo_name, int(t)


def _pair_stream(
    ta: TaskSource,
    tb: TaskSource,
    diff: tuple[int, str, str],
    family: str,
    max_per_demo: int,
    stride: int,
    rng: np.random.Generator,
) -> Iterator[ContrastivePair]:
    """Yield pairs for one instruction pairing, INTERLEAVING states from both tasks.

    Counterbalancing is not optional here, it is the experiment. The directional readout
    asks whether commanding the *non-demonstrated* instruction pushes the action away from
    what the demonstration did. If every state came from task A, that question is only ever
    asked in one direction, and any systematic asymmetry between the two instructions
    (one being closer to the policy's default behaviour, say) shows up as a spurious
    above- or below-chance score rather than cancelling out.

    An earlier version consumed `ta` fully before starting `tb`. Since callers take far
    fewer pairs than one task provides, `tb` was never reached and 100% of states came
    from A. Interleaving strictly alternates, so truncation at any n stays balanced.
    """
    span_idx, span_a, span_b = diff

    streams = [_states_from(source, max_per_demo, stride, rng) for source in (ta, tb)]
    alive = list(range(len(streams)))
    while alive:
        for k in list(alive):
            try:
                source, demo_name, t = next(streams[k])
            except StopIteration:
                alive.remove(k)
                continue
            yield ContrastivePair(
                pair_id=f"{ta.name}__{tb.name}__{source.name}__{demo_name}__t{t}",
                family=family,
                instruction_a=ta.instruction,
                instruction_b=tb.instruction,
                differing_span_a=span_a,
                differing_span_b=span_b,
                span_index=span_idx,
                source_task=source.name,
                source_file=f"{source.path.parent.name}/{source.path.name}",
                source_sha256=source.sha256,
                demo=demo_name,
                timestep=int(t),
                provenance=(
                    "state drawn from a pre-grasp timestep, so neither referent "
                    "has been manipulated and both instructions remain achievable"
                ),
                validation={
                    "single_referent_swap": True,
                    "shared_scene": True,
                    "pre_grasp": True,
                },
            )


# ------------------------------------------------------------------------- writing


def write_pairs(pairs: Sequence[ContrastivePair], out_path: Path, report: dict[str, Any]) -> str:
    """Write the JSONL stimulus file plus a sidecar manifest. Returns the content hash."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [json.dumps(p.as_dict(), sort_keys=True) for p in pairs]
    body = "\n".join(lines) + "\n"

    # newline="" suppresses Python's universal-newline translation, which on Windows
    # would rewrite every \n as \r\n. The hash below is taken over the LF form, so
    # without this the file on disk would not match its own recorded hash -- and the
    # stimulus set would hash differently depending on who generated it.
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)

    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "n_pairs": len(pairs),
        "content_sha256": content_hash,
        "report": report,
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return content_hash


def load_pairs(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_observation(
    pair: dict[str, Any], data_root: Path, chunk_size: int = 50
) -> dict[str, np.ndarray]:
    """Materialize the observation a pair refers to, plus the demonstrated action chunk.

    `action_chunk` is the demonstration's next `chunk_size` actions from this timestep --
    the trajectory the policy is being compared against. A policy predicts a whole chunk,
    so comparing it to the single action at time t (broadcast) would measure agreement
    with one instant rather than with the demonstrated behaviour, and would score a policy
    that stalls as highly as one that follows through.

    Short episodes are edge-padded with their final action, matching how a demonstration
    that has finished simply stops moving.
    """
    path = data_root / pair["source_file"]
    with h5py.File(path, "r") as f:
        demo = f["data"][pair["demo"]]
        t = pair["timestep"]
        actions = demo["actions"]
        end = min(t + chunk_size, actions.shape[0])
        chunk = actions[t:end]
        if chunk.shape[0] < chunk_size:
            pad = np.repeat(chunk[-1:], chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return {
            "agentview_rgb": demo["obs"]["agentview_rgb"][t],
            "eye_in_hand_rgb": demo["obs"]["eye_in_hand_rgb"][t],
            "joint_states": demo["obs"]["joint_states"][t],
            "gripper_states": demo["obs"]["gripper_states"][t],
            "ee_pos": demo["obs"]["ee_pos"][t],
            "ee_ori": demo["obs"]["ee_ori"][t],
            "action": actions[t],
            "action_chunk": chunk,
        }
