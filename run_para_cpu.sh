set -uo pipefail
cd "c:/Users/schma/vla-where-does-language-die"
PY=./.venv/Scripts/python.exe
for spec in "para_finetune:k1000dai/smolvla_libero_finetune" "para_scratch80k:k1000dai/smolvla_libero_scratch_80k"; do
  rid="${spec%%:*}"; ckpt="${spec##*:}"
  [ -f "results/$rid/metrics.json" ] && { echo "skip $rid"; continue; }
  echo "=== $(date -u +%H:%M:%S) $rid ==="
  $PY scripts/run_composition.py --checkpoint "$ckpt" --device cpu \
      --per-task 20 --fixed-state-control --phrasing paraphrase --run-id "$rid"
done
echo "=== PARAPHRASE COMPLETE ==="
