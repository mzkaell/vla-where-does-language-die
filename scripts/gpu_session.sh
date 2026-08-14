#!/usr/bin/env bash
# Unattended GPU session runner for an EPHEMERAL machine.
#
#   bash scripts/gpu_session.sh 2>&1 | tee -a gpu_session.log
#
# Designed around one fact: the machine is deleted permanently when the clock runs
# out, so anything not pushed is lost. Consequently:
#
#   * results are committed and pushed after EVERY stage, not at the end
#   * each stage is skipped if its output already exists, so the script is
#     resumable -- rerun it after any interruption and it continues
#   * `git pull --rebase` runs before each stage, so code pushed from elsewhere
#     mid-session is picked up without restarting
#   * failures do not abort the run: a broken stage is logged and the next one
#     starts, because partial results beat no results on a deadline
#
# Requires GIT_PUSH=1 and a credential helper or token already configured;
# without it the script still runs and just warns that nothing is being saved.

set -uo pipefail   # deliberately NOT -e: one failed stage must not kill the session

CKPT_A="k1000dai/smolvla_libero_finetune"
CKPT_B="k1000dai/smolvla_libero_scratch_80k"
DEVICE="${DEVICE:-cuda}"
TRIALS="${TRIALS:-40}"
PY="${PY:-python}"

say() { echo -e "\n=== [$(date -u +%H:%M:%S)] $* ==="; }

save() {
  # Commit and push whatever results exist. Called after every stage.
  local msg="$1"
  if [[ "${GIT_PUSH:-0}" != "1" ]]; then
    echo "!! GIT_PUSH not set -- results are NOT being saved off-machine"
    return 0
  fi
  git add -A results/ paper/ 2>/dev/null
  if git diff --cached --quiet; then echo "(nothing new to save)"; return 0; fi
  git -c user.name="${GIT_NAME:-gpu-session}" \
      -c user.email="${GIT_EMAIL:-gpu@session.local}" \
      commit -q -m "$msg" && git push -q origin HEAD && echo "pushed: $msg" \
      || echo "!! PUSH FAILED -- results exist locally but are NOT safe"
}

stage_done() { [[ -f "results/$1/metrics.json" ]]; }

say "GPU check"
nvidia-smi || echo "!! no nvidia-smi -- is this actually a GPU machine?"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 1

say "Data"
$PY scripts/download_data.py --all || echo "!! data download failed"

say "Stimuli"
[[ -f stimuli/libero_goal_pairs_v1.jsonl ]] || $PY scripts/build_pairs.py --suite libero_goal --n 400

# ----------------------------------------------------------------- smoke test
# Cheap, and it verifies the whole path on THEIR hardware before we spend hours.
say "Smoke test (few minutes)"
$PY scripts/run_localization.py --checkpoint "$CKPT_A" --device "$DEVICE" \
    --n-trials 2 --sites-limit 6 --null-sites 3 --min-trials 2 --resamples 500 \
    --run-id _gpu_smoke
if ! stage_done _gpu_smoke; then
  echo "!! SMOKE TEST FAILED -- stopping before burning GPU hours on a broken setup"
  exit 1
fi
rm -rf results/_gpu_smoke
echo "smoke test OK"

# ------------------------------------------------------------------- M2 sweep
for spec in "loc_finetune:$CKPT_A" "loc_scratch80k:$CKPT_B"; do
  rid="${spec%%:*}"; ckpt="${spec##*:}"
  if stage_done "$rid"; then echo "skip $rid (already done)"; continue; fi
  say "M2 localization: $rid"
  git pull --rebase -q 2>/dev/null
  $PY scripts/run_localization.py --checkpoint "$ckpt" --device "$DEVICE" \
      --n-trials "$TRIALS" --run-id "$rid"
  save "M2 localization sweep: $rid"
done

# ------------------------------------------------- proximity control (M2 control)
# Both arms trained, so there is no binding failure to recover -- only the network's
# ordinary causal structure. Subtracting this profile from the novel one cancels the
# artefact where recovery rises toward the output simply because late patches sit closer
# to it. Without this the M2 map is not interpretable.
for spec in "locctl_finetune:$CKPT_A" "locctl_scratch80k:$CKPT_B"; do
  rid="${spec%%:*}"; ckpt="${spec##*:}"
  if stage_done "$rid"; then echo "skip $rid (already done)"; continue; fi
  say "M2 proximity control: $rid"
  git pull --rebase -q 2>/dev/null
  $PY scripts/run_localization.py --checkpoint "$ckpt" --device "$DEVICE"       --n-trials "$TRIALS" --contrast-mode control --run-id "$rid"
  save "M2 proximity control: $rid"
done

# ----------------------------------------------------------- destination probe
# Answers the encoding-vs-readout question the patching sweep could not. Probing is
# correlational, so it is unaffected by the causal-proximity confound that made the sweep
# uninterpretable -- a different failure mode, which is the point.
if [[ -f scripts/run_probe.py ]]; then
  for spec in "probe_finetune:$CKPT_A" "probe_scratch80k:$CKPT_B"; do
    rid="${spec%%:*}"; ckpt="${spec##*:}"
    if stage_done "$rid"; then echo "skip $rid"; continue; fi
    say "Destination probe: $rid"
    git pull --rebase -q 2>/dev/null
    $PY scripts/run_probe.py --checkpoint "$ckpt" --device "$DEVICE"         --n-states 60 --run-id "$rid"
    save "Destination probe: $rid"
  done
fi

# --------------------------------------- position-resolved + M3 (if available)
# These are pushed from the laptop mid-session; the pull above picks them up.
git pull --rebase -q 2>/dev/null

if [[ -f scripts/run_position_patching.py ]]; then
  for spec in "pos_finetune:$CKPT_A" "pos_scratch80k:$CKPT_B"; do
    rid="${spec%%:*}"; ckpt="${spec##*:}"
    if stage_done "$rid"; then echo "skip $rid"; continue; fi
    say "Position-resolved patching: $rid"
    $PY scripts/run_position_patching.py --checkpoint "$ckpt" --device "$DEVICE" \
        --n-trials "$TRIALS" --run-id "$rid"
    save "Position-resolved patching: $rid"
  done
else
  echo "(scripts/run_position_patching.py not present yet -- skipping)"
fi

if [[ -f scripts/run_transplant.py ]]; then
  for spec in "transplant_finetune:$CKPT_A" "transplant_scratch80k:$CKPT_B"; do
    rid="${spec%%:*}"; ckpt="${spec##*:}"
    if stage_done "$rid"; then echo "skip $rid"; continue; fi
    say "M3 binding transplant: $rid"
    $PY scripts/run_transplant.py --checkpoint "$ckpt" --device "$DEVICE" \
        --n-trials "$TRIALS" --run-id "$rid"
    save "M3 binding transplant: $rid"
  done
else
  echo "(scripts/run_transplant.py not present yet -- skipping)"
fi

say "Session complete"
ls -1 results/
save "GPU session results"
echo "IMPORTANT: confirm on GitHub that results/ contains the new runs before the machine dies."
