#!/usr/bin/env bash
# Part 2 on Kaggle (GPU): reuse part 1 outputs (classifier + 7 SR runs), train A and B WITHOUT VGG
# (clean A -> B -> C chain), re-evaluate everything, draw the result figures, collect deliverables.
# Needs the output of notebook "notebooke14b9e1fed" (part 1) attached as an input.
set -euo pipefail

pip install -q medmnist lpips
python src/prepare_data.py --out data

PREV_A0=$(find /kaggle/input -path '*runs/sr/A0/G_final.pt' | head -n 1)
PREV_RUNS=$(dirname "$(dirname "$(dirname "$PREV_A0")")")
echo "part 1 runs: $PREV_RUNS"
mkdir -p runs/sr
cp -r "$PREV_RUNS/classifier" runs/
for r in A0 A_vgg B_vgg C_vgg C A0_degraded C_vgg_degraded; do
  mkdir -p "runs/sr/$r"
  cp "$PREV_RUNS/sr/$r/config.json" "$PREV_RUNS/sr/$r/G_final.pt" "$PREV_RUNS/sr/$r/train_summary.json" "runs/sr/$r/"
done

python src/train_sr.py --variant A --out runs/sr
python src/train_sr.py --variant B --out runs/sr

python src/evaluate.py --runs runs/sr/A0 runs/sr/A runs/sr/B runs/sr/C runs/sr/A_vgg runs/sr/B_vgg runs/sr/C_vgg \
  runs/sr/A0_degraded runs/sr/C_vgg_degraded --classifier runs/classifier --out results/eval --cross-degraded

python src/make_figures.py --ref A --improved B C C_vgg --best C_vgg

# deliverables: generator weights (small) + configs, next to the SR test images in outputs/sr_test
mkdir -p weights
for r in A0 A B C A_vgg B_vgg C_vgg A0_degraded C_vgg_degraded; do
  cp "runs/sr/$r/G_final.pt" "weights/G_$r.pt"
  cp "runs/sr/$r/config.json" "weights/G_$r.config.json"
done
cp runs/sr/A/train_summary.json results/train_summary_A.json
cp runs/sr/B/train_summary.json results/train_summary_B.json
rm -rf data/raw runs/sr/*/latest.pt
echo "PART2 DONE"
