"""Compositional grounding: does the policy compose familiar words into novel commands?

Why this test exists
--------------------
M0 and the conflict regime both came back null: SmolVLA followed destination instructions
at 75-95%. But every instruction we gave it was **one of the 10 tasks it was trained on**.
A policy can score perfectly on that by recognising which memorised sentence it heard --
sentence-level pattern matching -- without ever treating "bowl" and "rack" as separable
meanings it can recombine. Our design gave it no opportunity to fail.

This test removes the shortcut. All ten LIBERO-Goal scenes contain the same seven objects
(verified), so we can command combinations that are physically executable but were never
demonstrated:

    trained:  bowl -> plate, stove, cabinet-top     bottle -> rack, cabinet-top
    NOVEL:    bowl -> rack                          bottle -> plate, stove

Familiar object, familiar destination, novel pairing. If grounding is compositional, novel
commands should work as well as trained ones. If the policy is matching whole sentences,
novel commands should fall back to a memorised destination -- which is exactly the
object-word binding failure CLAUDE.md §3 predicts.

Reference-free readout
----------------------
A novel composition has no demonstration, so the demo-trajectory reference used elsewhere
does not exist. Instead we ask **which way the arm actually goes**.

Each destination has an *anchor*: the mean end-effector position at the end of the
demonstrations that finish there. Anchors are stable (sd 0.03-0.05 m) and well separated
(0.2-0.5 m), and the cabinet-top anchor agrees to within 0.07 m whether the object was the
bowl or the bottle -- so it tracks the destination, not the task.

Given a state and an instruction we integrate the predicted delta-EE actions into a net
displacement, then score which anchor direction it best matches. Cosine is scale-free, so
the unknown action-scaling of the controller does not matter.

THE CONTROL THAT MAKES THIS INTERPRETABLE: trained compositions must score high. If they
do not, the readout is broken and nothing can be concluded about novel ones. That check
runs in the same pass and is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Destination phrase -> the tasks whose demonstrations end at that destination.
DESTINATIONS: dict[str, list[str]] = {
    "the plate": ["put the bowl on the plate"],
    "the stove": ["put the bowl on the stove"],
    "top of the cabinet": [
        "put the bowl on top of the cabinet",
        "put the wine bottle on top of the cabinet",
    ],
    "the rack": ["put the wine bottle on the rack"],
}

# object phrase -> destinations it was trained with
TRAINED: dict[str, set[str]] = {
    "bowl": {"the plate", "the stove", "top of the cabinet"},
    "wine bottle": {"the rack", "top of the cabinet"},
}

# Which demo files supply post-grasp states for each object.
OBJECT_SOURCE_TASKS: dict[str, list[str]] = {
    "bowl": [
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
        "put_the_bowl_on_top_of_the_cabinet",
    ],
    "wine bottle": [
        "put_the_wine_bottle_on_the_rack",
        "put_the_wine_bottle_on_top_of_the_cabinet",
    ],
}


def instruction_for(obj: str, destination: str) -> str:
    return f"put the {obj} on {destination}"


def is_novel(obj: str, destination: str) -> bool:
    return destination not in TRAINED[obj]


@dataclass
class CompositionOutcome:
    """One (state, instruction) trial."""

    trial_id: str
    obj: str
    named_destination: str
    novel: bool
    chosen_destination: str
    correct: bool
    cosines: dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    """Cosine of the named destination minus the best competing one. <0 means it lost."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "object": self.obj,
            "named_destination": self.named_destination,
            "novel": self.novel,
            "chosen_destination": self.chosen_destination,
            "correct": self.correct,
            "margin": self.margin,
            "cosines": self.cosines,
        }


def score_direction(
    trial_id: str,
    obj: str,
    named_destination: str,
    ee_pos: np.ndarray,
    predicted_delta: np.ndarray,
    anchors: dict[str, np.ndarray],
    min_displacement: float = 1e-6,
) -> CompositionOutcome | None:
    """Which destination does the predicted motion head toward?

    `predicted_delta` is the net (dx, dy, dz) the policy commands over the chunk.
    Returns None when the commanded motion is too small to have a direction, which would
    otherwise contribute an arbitrary argmax and quietly bias the accuracy.
    """
    norm = float(np.linalg.norm(predicted_delta))
    if norm < min_displacement:
        return None

    unit_pred = predicted_delta / norm
    cosines: dict[str, float] = {}
    for dest, anchor in anchors.items():
        to_anchor = anchor - ee_pos
        d = float(np.linalg.norm(to_anchor))
        if d < 1e-9:
            continue
        cosines[dest] = float(np.dot(unit_pred, to_anchor / d))

    if not cosines:
        return None

    chosen = max(cosines, key=lambda k: cosines[k])
    others = [v for k, v in cosines.items() if k != named_destination]
    margin = cosines.get(named_destination, -1.0) - (max(others) if others else -1.0)

    return CompositionOutcome(
        trial_id=trial_id,
        obj=obj,
        named_destination=named_destination,
        novel=is_novel(obj, named_destination),
        chosen_destination=chosen,
        correct=chosen == named_destination,
        cosines=cosines,
        margin=margin,
    )


def build_anchors(task_endpoints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Destination anchors: mean final end-effector position over the tasks ending there."""
    anchors: dict[str, np.ndarray] = {}
    for dest, tasks in DESTINATIONS.items():
        pts = [task_endpoints[t] for t in tasks if t in task_endpoints]
        if pts:
            anchors[dest] = np.mean(pts, axis=0)
    return anchors


def aggregate(outcomes: list[CompositionOutcome], resamples: int = 10_000, seed: int = 0):
    """Accuracy for trained vs novel compositions, with bootstrap CIs."""
    from src.eval.stats import bootstrap_mean

    def block(sel: list[CompositionOutcome]) -> dict[str, Any] | None:
        if len(sel) < 5:
            return None
        acc = np.array([float(o.correct) for o in sel])
        marg = np.array([o.margin for o in sel])
        return {
            "n": len(sel),
            "accuracy": bootstrap_mean(acc, resamples=resamples, seed=seed).as_dict(),
            "margin": bootstrap_mean(marg, resamples=resamples, seed=seed).as_dict(),
        }

    trained = [o for o in outcomes if not o.novel]
    novel = [o for o in outcomes if o.novel]

    # Where do novel commands go when they are wrong? If the policy is matching whole
    # sentences, errors should land on a destination the object WAS trained with.
    fallback: dict[str, int] = {}
    for o in novel:
        if not o.correct:
            memorised = o.chosen_destination in TRAINED[o.obj]
            key = o.chosen_destination + (" [trained for this object]" if memorised else "")
            fallback[key] = fallback.get(key, 0) + 1

    # PRIMARY ANALYSIS: within-destination, across objects.
    #
    # The raw trained-vs-novel gap is confounded, badly. The policy has a strong prior over
    # destinations independent of what was asked (it heads for the cabinet most of the
    # time), and per-destination accuracy ranges from ~0.1 at the rack to ~0.95 at the
    # cabinet. Since the trained and novel sets contain different mixes of destinations,
    # an aggregate difference between them partly measures that mix rather than
    # composition.
    #
    # Holding the destination fixed removes it. For destination D, compare trials where D
    # was trained for the named object against trials where it was not. Same destination
    # word, same anchor, same target direction -- only the object and the trained/novel
    # status change.
    within: dict[str, Any] = {}
    for dest in DESTINATIONS:
        tr = [o for o in trained if o.named_destination == dest]
        nv = [o for o in novel if o.named_destination == dest]
        if len(tr) < 5 or len(nv) < 5:
            continue
        t_acc = float(np.mean([o.correct for o in tr]))
        n_acc = float(np.mean([o.correct for o in nv]))
        within[dest] = {
            "trained": {"n": len(tr), "accuracy": t_acc, "objects": sorted({o.obj for o in tr})},
            "novel": {"n": len(nv), "accuracy": n_acc, "objects": sorted({o.obj for o in nv})},
            "gap": t_acc - n_acc,
        }

    n_dest = len(DESTINATIONS)
    return {
        "within_destination": within,
        "note": (
            "within_destination is the PRIMARY comparison. The aggregate trained/novel "
            "blocks below are confounded by an object-independent destination prior."
        ),
        "chance_level": 1.0 / n_dest,
        "n_destinations": n_dest,
        "trained": block(trained),
        "novel": block(novel),
        "novel_error_destinations": fallback,
        "by_object": {
            obj: {
                "trained": block([o for o in trained if o.obj == obj]),
                "novel": block([o for o in novel if o.obj == obj]),
            }
            for obj in sorted({o.obj for o in outcomes})
        },
    }


def demo_of(trial_id: str) -> str:
    """Cluster label for the bootstrap: the demonstration a trial's state came from.

    trial_id is `task__demo__t<step>__object__destination`, so the first two fields
    identify the episode. States from one episode share a scene and object placement and
    sit at adjacent timesteps, so they are not independent observations.
    """
    parts = trial_id.split("__")
    return "__".join(parts[:2]) if len(parts) >= 2 else trial_id


def substitution_excess(
    outcomes: list[dict[str, Any]],
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """How often novel-command errors land on a trained destination, ABOVE CHANCE.

    The raw rate is not interpretable on its own, and reporting it alone overstates the
    result badly. For `bowl -> the rack`, bowl was trained on the other three
    destinations, so *every* possible wrong answer is a trained one and the raw rate is
    100% by construction. Pooled across our cells the chance floor is ~0.83, against a raw
    rate of ~0.94 -- an excess of about 0.10, not the 0.94 the raw number suggests.

    Per trial we compute the fraction of the wrong answers that are trained for that
    object, which is that trial's chance level, and report observed minus chance with a
    demonstration-clustered CI.
    """
    from src.eval.stats import bootstrap_mean

    errs = [o for o in outcomes if o.get("novel") and not o.get("correct")]
    if len(errs) < 5:
        return {"n_errors": len(errs), "insufficient": True}

    dests = list(DESTINATIONS)
    hit, chance, groups = [], [], []
    for o in errs:
        trained = TRAINED[o["object"]]
        wrong = [d for d in dests if d != o["named_destination"]]
        hit.append(float(o["chosen_destination"] in trained))
        chance.append(len([d for d in wrong if d in trained]) / len(wrong))
        groups.append(demo_of(o["trial_id"]))

    hit_a, ch_a = np.array(hit), np.array(chance)
    excess = bootstrap_mean(hit_a - ch_a, resamples=resamples, seed=seed, cluster=groups)
    return {
        "n_errors": len(errs),
        "observed_rate": float(hit_a.mean()),
        "chance_rate": float(ch_a.mean()),
        "excess_over_chance": excess.as_dict(),
        "cells": {
            f"{o['object']} told '{o['named_destination']}'": round(c, 2)
            for o, c in zip(errs, chance, strict=False)
        },
    }


def named_destination_sensitivity(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the choice on NOVEL commands depend on which destination was named?

    This is the test that separates a compositional account from a fixed
    object-conditioned prior. A policy that has simply learned "object X goes to A" would
    produce the same distribution of chosen destinations no matter what it was told; a
    policy doing partial composition shifts toward whatever was named.

    Returns, per (object, named destination), the distribution over chosen destinations,
    plus the diagonal rate -- how often the named destination was chosen -- which is the
    single number that discriminates the two accounts.
    """
    table: dict[str, dict[str, float]] = {}
    diagonal: dict[str, float] = {}
    for obj in sorted({o["object"] for o in outcomes if o.get("novel")}):
        for named in sorted(
            {o["named_destination"] for o in outcomes if o.get("novel") and o["object"] == obj}
        ):
            sel = [
                o for o in outcomes
                if o.get("novel") and o["object"] == obj and o["named_destination"] == named
            ]
            if not sel:
                continue
            counts: dict[str, int] = {}
            for o in sel:
                counts[o["chosen_destination"]] = counts.get(o["chosen_destination"], 0) + 1
            key = f"{obj} told '{named}'"
            table[key] = {k: v / len(sel) for k, v in sorted(counts.items(), key=lambda x: -x[1])}
            diagonal[key] = counts.get(named, 0) / len(sel)
    return {"distribution": table, "chose_the_named_destination": diagonal}
