#!/usr/bin/env bash
# Unattended pipeline: dataset generation -> NN surrogate training -> surrogate-assisted EA
# -> pure-AVL EA cross-check. Each stage pings ntfy, so your phone tracks progress.
#
# Launch detached (survives closing the SSH session):
#     nohup bash scripts/run_overnight.sh >/dev/null 2>&1 &
# Follow live if you want:
#     tail -f runs/overnight_*.log
#
# Notes:
# - Regenerates data/dataset.csv from scratch on purpose: objective changes make old fitness
#   labels stale, and this script is the "rebuild everything" path.
# - nohup shells don't read ~/.bashrc, so AVL_BIN is re-derived from vendor/ if unset.
set -uo pipefail   # deliberately no -e: failures are caught so a ntfy alert can be sent

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="runs/overnight_${STAMP}.log"
mkdir -p runs
exec >>"$LOG" 2>&1

NTFY_TOPIC="${NTFY_TOPIC:-avlnn-pipeline}"   # ntfy.sh topics are public -- set your own
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"

notify() {  # $1 = title, $2 = message, $3 = tags (optional)
  curl -fsS -m 10 -H "Title: $1" -H "Tags: ${3:-robot}" -d "$2" \
    "$NTFY_SERVER/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fail() {
  echo "!! FAILED at stage: $1 ($(date))"
  notify "Overnight pipeline FAILED" "Stage: $1
Log: $LOG
$(tail -n 5 "$LOG")" "rotating_light"
  exit 1
}

source .venv/bin/activate || fail "venv activation (.venv missing?)"

if [ -z "${AVL_BIN:-}" ]; then
  AVL_BIN="$(ls "$REPO_ROOT"/vendor/*/bin/avl 2>/dev/null | head -n 1)"
  export AVL_BIN
fi
[ -n "${AVL_BIN:-}" ] && [ -x "$AVL_BIN" ] || fail "AVL binary not found (AVL_BIN unset, nothing in vendor/)"

WORKERS="$(nproc)"
T0=$SECONDS
echo "=== Overnight pipeline started $(date)  workers=$WORKERS  AVL_BIN=$AVL_BIN"
notify "Overnight pipeline started" \
  "1) dataset  2) NN training  3) surrogate EA  4) pure-AVL cross-check" "rocket"

echo "=== [1/4] Dataset: 4000 LHS samples via real AVL ($(date))"
python scripts/run_dataset_gen.py --samples 4000 --workers "$WORKERS" --seed 0 \
  --out data/dataset.csv || fail "1/4 dataset generation"
notify "1/4 dataset done" "$(grep -o 'Generated .*' "$LOG" | tail -n 1)" "white_check_mark"

echo "=== [2/4] Training the NN surrogate ($(date))"
python scripts/train_surrogate.py --dataset data/dataset.csv --out data/surrogate.pt \
  || fail "2/4 surrogate training"
notify "2/4 NN trained" "$(grep -o 'Validation MSE.*' "$LOG" | tail -n 1)" "white_check_mark"

echo "=== [3/4] Surrogate-assisted EA ($(date))"
python scripts/run_surrogate_ea.py --population 240 --generations 600 --seed 0 \
  --revalidate-top 6 || fail "3/4 surrogate EA"
notify "3/4 surrogate EA done" \
  "$(grep -o 'Best real-AVL-confirmed fitness.*' "$LOG" | tail -n 1)" "white_check_mark"

echo "=== [4/4] Pure-AVL EA cross-check ($(date))"
python scripts/run_ea.py --population 40 --generations 60 --workers "$WORKERS" --seed 10 \
  --ntfy-topic '' || fail "4/4 pure-AVL EA"

ELAPSED_MIN=$(( (SECONDS - T0) / 60 ))
SUMMARY="$(python - <<'PY'
import json
s = json.load(open("runs/surrogate_ea_result.json"))
p = json.load(open("runs/ea_result.json"))
sf = s.get("best_real_fitness")
print(f"surrogate EA (real-AVL): {sf:.4f}" if sf is not None else
      "surrogate EA: NO feasible re-validated design!")
print(f"pure-AVL EA cross-check: {p['best_fitness']:.4f}")
PY
)"
echo "=== All stages done in ${ELAPSED_MIN} min"
echo "$SUMMARY"
notify "Overnight pipeline COMPLETE (${ELAPSED_MIN} min)" "$SUMMARY
Results: runs/surrogate_ea_result.json + runs/ea_result.json" "tada"
