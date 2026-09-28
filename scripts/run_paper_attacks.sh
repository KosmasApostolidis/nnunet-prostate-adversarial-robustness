#!/usr/bin/env bash
# Reproduce Table I and the slice-level analyses: 4 arms (noise, FGSM, PGD, Auto-PGD)
# x 2 models x 5 folds, each run natively at every epsilon, then aggregate.
# Usage: PARALLEL=4 bash scripts/run_paper_attacks.sh   (from the repository root)
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-results/paper}
mkdir -p "$OUT/logs"
job() {
  local a=$1 m=$2 f=$3
  PYTHONPATH=src python experiments/slice_vulnerability_analysis.py --model "$m" --fold "$f" \
    --attack "$a" --seed 42 --output-dir "$OUT/$a" > "$OUT/logs/${a}_${m}_fold${f}.log" 2>&1
  echo "${a} ${m} fold${f} rc=$?"
}
export -f job; export OUT
for a in auto_pgd pgd fgsm noise; do for m in wg zones; do for f in 0 1 2 3 4; do echo "$a $m $f"; done; done; done \
  | xargs -P "${PARALLEL:-4}" -L 1 bash -c 'job "$0" "$1" "$2"'
PYTHONPATH=src python experiments/camera_ready_analysis.py --root "$OUT" --arms fgsm noise pgd auto_pgd

# Prediction flip (adversarial vs clean prediction), FGSM and noise.
for a in fgsm noise; do for m in wg zones; do for f in 0 1 2 3 4; do echo "$a $m $f"; done; done; done \
  | xargs -P "${PARALLEL:-4}" -L 1 bash -c 'PYTHONPATH=src python experiments/prediction_flip_analysis.py \
      --attack "$0" --model "$1" --fold "$2" --seed 42 --output-dir "$OUT/prediction_flip"'

# Slice-level statistics (adjustment, FGSM - noise, label source, ASD change, flip).
PYTHONPATH=src python experiments/slice_statistics.py --root "$OUT" --flip-root "$OUT/prediction_flip"
