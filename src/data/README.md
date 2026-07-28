# Contrastive stimulus set — schema

Released artifact for M1. Built by `scripts/build_pairs.py`, written to
`stimuli/<suite>_pairs_v1.jsonl` with a sidecar `.manifest.json`.

## What a pair is

One visual state plus two instructions differing in **exactly one referent**, where both
instructions are individually achievable from that state. Running the policy on identical
pixels under instruction A and instruction B isolates the causal contribution of the
swapped word — nothing else differs, so any change in the action is attributable to it.

LIBERO-Goal is used because its ten tasks share one kitchen scene with a fixed object set
and only the commanded goal changes. Vision therefore underdetermines the target and the
instruction must be read. LIBERO-Object and -Spatial change the scene along with the task,
which would confound the referent swap with a visual difference.

## Why references, not pixels

Each record points at `(source_file, demo, timestep)` inside the public LIBERO
demonstration HDF5s rather than embedding images. The stimulus file stays small and
releasable, and `source_sha256` pins the exact source bytes. Use
`src.data.build_pairs.load_observation(pair, data_root)` to materialize observations.

## Record fields

| Field | Type | Meaning |
|---|---|---|
| `pair_id` | str | Unique, deterministic: `taskA__taskB__source__demo__t<step>` |
| `family` | str | `destination_swap` or `object_swap` |
| `instruction_a` / `instruction_b` | str | The two instructions |
| `differing_span_a` / `differing_span_b` | str | The swapped referent on each side |
| `span_index` | int | Word index where the swap begins |
| `source_task` | str | Which task's demo the state came from |
| `source_file` | str | Relative path, e.g. `libero_goal/put_the_bowl_on_the_plate_demo.hdf5` |
| `source_sha256` | str | Hash of the source HDF5 |
| `demo` | str | e.g. `demo_17` |
| `timestep` | int | Index within the episode |
| `provenance` | str | Why this state qualifies |
| `validation` | obj | Which gates passed |

States are drawn from **both** tasks in a pairing, so the set is not biased toward scenes
where one instruction happens to be the demonstrated one.

## Validity gates

A pair ships only if all pass.

**1. Exactly one referent differs.** Enforced by `single_span_diff`, a word-level diff
requiring exactly one edit block. Two subtleties, both of which silently admit invalid
pairs if handled naively:

- *Trimming the common prefix and suffix is not sufficient.* `"put the bowl on the plate"`
  vs `"put the mug on the stove"` has common prefix `"put the"` and no common suffix, so
  trimming reports one span — but that is two independent swaps sharing the words
  `"on the"` between them. A real diff sees two blocks and rejects it.
- *Blocks separated only by filler words must be merged.* `"the rack"` vs
  `"top of the cabinet"` diffs into two blocks around the shared `"the"`, though it is one
  referent. Blocks separated only by `{the, a, an, of}` are merged back.

  That merge alone is too permissive: `"put the cream cheese in the bowl"` vs
  `"put the wine bottle on the rack"` changes object *and* destination, yet its two blocks
  are also separated by `"the"`. So a merged span must additionally contain no
  preposition — a referent is a noun phrase. This case is a regression test.

**2. Shared scene.** Guaranteed by construction within LIBERO-Goal.

**3. Both actions individually valid.** States are taken from **pre-grasp** timesteps —
before the first gripper-close command in the episode. Nothing has been grasped yet, so
every task in the shared scene remains achievable. After the grasp the trajectory has
committed and the alternative instruction is no longer reachable without regrasping.

## Current release

`stimuli/libero_goal_pairs_v1.jsonl` — 200 pairs, schema `libero-goal-pairs/1`.

Minimal pairings found among the 6 downloaded tasks (10 candidate pairings rejected):

| Family | Swap |
|---|---|
| destination | `plate` → `stove` |
| destination | `the plate` → `top of the cabinet` |
| destination | `the stove` → `top of the cabinet` |
| destination | `the rack` → `top of the cabinet` |
| object | `bowl` → `wine bottle` |

Counts: 160 destination swaps, 40 object swaps. The object-swap arm is thinner because
only one object pair shares a destination; downloading the remaining four LIBERO-Goal task
files would add more.

## Reproducing

```bash
python scripts/build_pairs.py --suite libero_goal --n 200
```

Deterministic given `--seed`. The manifest records the content hash, the accepted and
rejected pairings, and counts by family.
