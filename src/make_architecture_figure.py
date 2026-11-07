"""Annotated architecture diagram (generator + discriminator + losses) for the report.

Usage: python src/make_architecture_figure.py --out figures/architecture.png
Shapes are for a 32x32 input, x4 scale, 64 channels, 16 residual blocks (train_sr.py defaults).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK_2, LINE = "#0b0b0b", "#52514e", "#8a8984"
FILL = {"io": "#eef2f7", "conv": "#dbe8f8", "res": "#c9dcf5", "attn": "#fde3cf", "up": "#d7efe3",
        "disc": "#ece6f6", "loss": "#fff4cc", "frozen": "#eeeeee"}


def box(ax, x, y, w, h, text, kind, fs=8, bold=False, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=FILL[kind], ec=INK_2, lw=0.9, ls="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1, text=None, dy=0.12, style="-|>", color=LINE, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, lw=1,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8, color=INK_2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/architecture.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

    # ---------------- generator (top row)
    ax.text(0.15, 5.95, "Generator G — SRResNet / SRGAN (trained from scratch, no pretrained weights)",
            fontsize=10, fontweight="bold", color=INK)
    y, h = 4.35, 1.05
    box(ax, 0.15, y, 0.95, h, "LR input\n3×32×32\n[−1, 1]", "io")
    box(ax, 1.35, y, 1.0, h, "Conv 9×9\n64 ch\nPReLU", "conv")
    rb = box(ax, 2.65, y - 0.15, 3.3, h + 0.3, "", "res")
    ax.text(4.3, y + h + 0.03, "16 × residual block", ha="center", fontsize=8, fontweight="bold", color=INK)
    box(ax, 2.8, y + 0.1, 0.85, 0.85, "Conv 3×3\nBN\nPReLU", "conv", fs=7)
    box(ax, 3.8, y + 0.1, 0.85, 0.85, "Conv 3×3\nBN", "conv", fs=7)
    box(ax, 4.8, y + 0.1, 1.0, 0.85, "Channel\nattention (SE)\nB, C only", "attn", fs=6.8, dashed=True)
    arrow(ax, 3.65, y + 0.52, 3.8, y + 0.52); arrow(ax, 4.65, y + 0.52, 4.8, y + 0.52)
    ax.text(4.3, y - 0.12, "x + f(x)   (skip inside every block)", ha="center", fontsize=6.8, color=INK_2)
    box(ax, 6.25, y, 1.0, h, "Conv 3×3\nBN\n+ global skip", "conv", fs=7.5)
    box(ax, 7.5, y, 1.4, h, "Upsample ×2, twice\nConv 3×3\n→ PixelShuffle → PReLU", "up", fs=6.8)
    box(ax, 9.15, y, 0.9, h, "Conv 9×9\ntanh", "conv")
    box(ax, 10.3, y, 1.05, h, "SR output\n3×128×128", "io", bold=True)
    for x0, x1 in ((1.1, 1.35), (2.35, 2.65), (5.95, 6.25), (7.25, 7.5), (8.9, 9.15), (10.05, 10.3)):
        arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(6.75, y - 0.32, "32×32 → 64×64 → 128×128", ha="center", fontsize=6.8, color=INK_2)
    ax.text(11.5, y + 0.5, "1.55 M params\n(1.56 M with SE)\n≈2.6–2.9 ms / image\n(T4, batch 64)",
            fontsize=6.8, color=INK_2, va="center")

    # ---------------- losses (middle)
    ax.text(0.15, 3.55, "Training losses, all computed between SR and HR (same budget for every variant: 40 epochs, 8,720 generator updates)",
            fontsize=10, fontweight="bold", color=INK)
    ly, lh = 2.45, 0.85
    box(ax, 0.15, ly, 1.75, lh, "L1 pixel loss\nall variants (A0 = L1 only)", "loss", fs=7.5)
    box(ax, 2.05, ly, 1.95, lh, "Adversarial loss  ×1e-3\nA, B, C — after 25 %\nL1-only warm-up", "loss", fs=7.2)
    box(ax, 4.15, ly, 2.15, lh, "Structure loss (C)\n0.1·L1(Sobel SR, Sobel HR)\n+ 0.1·(1 − SSIM)", "loss", fs=7.2)
    box(ax, 6.45, ly, 2.25, lh, "VGG19 relu5_4 perceptual ×6e-3\nfrozen, ImageNet — declared\n& ablated (*_vgg runs)",
        "frozen", fs=7, dashed=True)
    box(ax, 8.85, ly, 2.5, lh, "Not a loss — note: the evaluation\nResNet-18 is never used in training\n(avoids optimising for the evaluator)",
        "frozen", fs=7)

    # ---------------- discriminator (bottom)
    ax.text(0.15, 1.85, "Discriminator D — SRGAN style (A, B, C only)", fontsize=10, fontweight="bold", color=INK)
    dy, dh = 0.45, 1.0
    box(ax, 0.15, dy, 1.15, dh, "HR or SR\n3×128×128", "io")
    box(ax, 1.55, dy, 4.3, dh, "8 × [Conv 3×3 → BN → LeakyReLU 0.2]\nchannels 64 → 512, stride 2 every other layer\n128 → 8 px",
        "disc", fs=7.5)
    box(ax, 6.1, dy, 1.6, dh, "Global avg pool\nConv 1×1 → 1024\nLeakyReLU", "disc", fs=7.5)
    box(ax, 7.95, dy, 1.25, dh, "Conv 1×1 → 1\nreal / fake\nlogit", "disc", fs=7.5)
    box(ax, 9.45, dy, 1.9, dh, "BCE with logits\n5.2 M params", "loss", fs=7.5)
    for x0, x1 in ((1.3, 1.55), (5.85, 6.1), (7.7, 7.95), (9.2, 9.45)):
        arrow(ax, x0, dy + dh / 2, x1, dy + dh / 2)

    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved", args.out)


if __name__ == "__main__":
    main()
