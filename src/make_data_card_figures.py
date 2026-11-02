"""Figures for the data card (fiche de données).

Usage:
    python src/make_data_card_figures.py --data data/processed --out figures/data_card

Only the TRAIN split is shown, so no test image ever appears in a figure.
Produces:
    fig1_class_distribution.png   train images per class (imbalance)
    fig2_examples_per_class.png   random real 128 px examples, one row per class
    fig3_degradations.png         HR 128 | LR bicubic 32 | LR blur+noise 32 | bicubic x4 upsampling
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SEED = 0
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"

SHORT = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}

plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK, "axes.edgecolor": GRID,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def read_counts(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [(int(r["class_id"]), int(r["train"]), float(r["train_pct"])) for r in rows]


def fig_class_distribution(counts, out):
    counts = sorted(counts, key=lambda r: r[1])  # smallest at the bottom of the reading order
    names = [SHORT[c] for c, _, _ in counts]
    vals = [n for _, n, _ in counts]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.barh(names, vals, color=BAR, height=0.62)
    ax.set_xlabel("Training images")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    xmax = max(vals)
    ax.set_xlim(0, xmax * 1.18)
    for b, (_, n, pct) in zip(bars, counts):
        ax.text(b.get_width() + xmax * 0.01, b.get_y() + b.get_height() / 2,
                f"{n:,}  ({pct:.1f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_title("DermaMNIST train split: images per class", loc="left", fontsize=11, color=INK)
    fig.savefig(out)
    plt.close(fig)


def fig_examples(hr, labels, out, per_class=6):
    rng = np.random.default_rng(SEED)
    classes = sorted(SHORT)
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.25, len(classes) * 1.3))
    for r, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for k in range(per_class):
            ax = axes[r, k]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if k < len(pick):
                ax.imshow(hr[pick[k]])
        axes[r, 0].set_ylabel(SHORT[c], rotation=0, ha="right", va="center", fontsize=9, color=INK)
    fig.suptitle("Real 128 px training examples (random, seed 0)", x=0.02, ha="left", fontsize=11)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(out)
    plt.close(fig)


def fig_degradations(d, out, n=4):
    rng = np.random.default_rng(SEED + 1)
    labels = d["labels"]
    # one image from n different classes, to show variety
    classes = rng.choice(sorted(SHORT), size=n, replace=False)
    picks = [rng.choice(np.flatnonzero(labels == c)) for c in classes]
    cols = ["HR 128 px\n(target)", "LR bicubic\n32 px", "LR blur + noise\n32 px", "Bicubic ×4\nupsampling"]
    fig, axes = plt.subplots(n, 4, figsize=(4 * 1.7, n * 1.75))
    for r, i in enumerate(picks):
        up = np.asarray(Image.fromarray(d["lr_bicubic"][i]).resize((128, 128), Image.BICUBIC))
        imgs = [d["hr"][i], d["lr_bicubic"][i], d["lr_degraded"][i], up]
        for k, img in enumerate(imgs):
            ax = axes[r, k]
            # LR shown with nearest neighbour so the 32 px pixels stay visible
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[k], fontsize=9, color=INK)
        axes[r, 0].set_ylabel(SHORT[int(labels[i])], rotation=0, ha="right", va="center",
                              fontsize=9, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="figures/data_card")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = np.load(os.path.join(args.data, "train.npz"))
    fig_class_distribution(read_counts(os.path.join(args.splits, "class_counts.csv")),
                           os.path.join(args.out, "fig1_class_distribution.png"))
    fig_examples(train["hr"], train["labels"], os.path.join(args.out, "fig2_examples_per_class.png"))
    fig_degradations(train, os.path.join(args.out, "fig3_degradations.png"))
    print("Saved figures to", args.out)


if __name__ == "__main__":
    main()
