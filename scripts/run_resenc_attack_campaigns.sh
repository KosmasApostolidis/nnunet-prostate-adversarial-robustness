#!/usr/bin/env bash
# Full attack campaigns against the three ResEnc checkpoints.
#
# Mirrors the published unet campaigns arm-for-arm and protocol-for-protocol
# (native per-epsilon, 20 steps, seed 42, folds 0-4) so a ResEnc column can be
# compared against its unet counterpart. auto_pgd_plus is omitted: CLAUDE.md
# records it as a refuted negative result that is weaker than auto_pgd.
#
# Zones runs first for all three architectures because it is the cheap half
# (~6 GPU-h per architecture against ~89 for whole gland), so a complete zones
# picture lands hours before the whole-gland one.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

# The thirteen arms of main_comparison_wg_boxplots.png, the figure this
# campaign extends to the ResEnc checkpoints. Both models run the same list so
# the two datasets stay directly comparable. Excluded on purpose: a_pgd and
# apgd_updated (in the whole-gland tree but not in that figure) and
# auto_pgd_plus (a refuted negative result, weaker than auto_pgd).
ARMS="fgsm pgd auto_pgd segpgd cospgd dag sea blade1 blade blade_mm auto_pgd_r3 blade_boundary blade_frontier"
WG_ARMS="$ARMS"
ZONES_ARMS="$ARMS"
# Concurrency is per model because the two networks differ in footprint: a
# zones job holds ~4.3 GB, a whole-gland job ~8 GB. A single job leaves the GPU
# badly underused, so the wall-clock win comes from overlapping jobs. These
# values leave headroom on the 143 GB card for the edge-attribution shards
# sharing it, whose footprint swings between 1.5 and 42 GB.
ZONES_JOBS="${ZONES_JOBS:-12}"
WG_JOBS="${WG_JOBS:-8}"
LOGDIR=logs/resenc_campaign

run_one() {
  local model=$1 arch=$2 arms=$3 outdir=$4
  local log="$LOGDIR/${model}_${arch}.log"
  local JOBS
  if [ "$model" = "zones" ]; then JOBS="$ZONES_JOBS"; else JOBS="$WG_JOBS"; fi
  echo "=== $(date +%H:%M) START $model $arch -> $outdir" | tee -a "$LOGDIR/campaign.log"
  python experiments/run_twelve_arm_wg_campaign.py \
    --model "$model" --arch "$arch" --native-per-eps --resume \
    --jobs "$JOBS" --arms $arms --output-dir "$outdir" >> "$log" 2>&1
  local rc=$?
  echo "=== $(date +%H:%M) END   $model $arch rc=$rc" | tee -a "$LOGDIR/campaign.log"
  return $rc
}

for arch in resenc_m resenc_l resenc_xl; do
  run_one zones "$arch" "$ZONES_ARMS" "results/blade_attack/zones/campaign_${arch}"
done
for arch in resenc_m resenc_l resenc_xl; do
  run_one wg "$arch" "$WG_ARMS" "results/blade_attack/whole_gland/native_per_eps_${arch}"
done
echo "=== $(date +%H:%M) ALL CAMPAIGNS FINISHED" | tee -a "$LOGDIR/campaign.log"
