"""Result figures for the report (run after evaluate.py).

Reads   data/processed/test.npz, outputs/sr_test/<method>.npz, results/eval/per_image/<method>.npz,
        results/eval/metrics.json
Writes  figures/results/
    fig4_comparison_zoom.png   LR | bicubic | A0 | A | B | C | HR, with zooms on the lesion border
    fig5_error_maps.png        |SR - HR| and contour (Sobel) error maps
    fig6_failures.png          predictions flipped by SR, and strongest hallucinated contours
    fig7_metrics.png           macro F1 per method; distortion (PSNR) vs utility (F1)
    fig8_per_class_recall.png  per-class recall heatmap
    failure_summary.json       counts of prediction flips per method
    figures_b64.json           all PNGs base64-encoded in one text file (easy to copy off Kaggle)

Only TEST images are shown here, after all training and model selection were finished.

Usage:
    python src/make_figures.py --best C_vgg --ref A --improved B C
"""
import argparse
import base64
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_otsu

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BAR, ACCENT = "#2a78d6", "#c2410c"
CLASS_SHORT = ["Act. keratosis", "Basal cell carc.", "Benign keratosis", "Dermatofibroma",
               "Melanoma", "Nevus", "Vascular"]
LABEL = {"bicubic": "Bicubic", "A0": "A0 · L1", "A": "A · SRGAN", "A_vgg": "A · SRGAN+VGG",
         "B": "B · +attention", "B_vgg": "B+VGG", "C": "C · +structure", "C_vgg": "C+VGG",
         "real_HR": "Real HR"}
ZOOM = 32  # zoom window on the 128 px image

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                     "ytick.color": INK_2, "axes.edgecolor": GRID, "figure.dpi": 120,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def lum(x):
    x = x.astype(np.float32) / 255.0
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def sobel(g):
    return np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))


def zoom_box(hr):
    """Top-left corner of a ZOOM x ZOOM window centred on the strongest point of the lesion border."""
    g = lum(hr)
    mask = g < threshold_otsu(g)
    border = mask ^ ndimage.binary_erosion(mask, iterations=2)
    score = sobel(g) * border
    h = ZOOM // 2
    score[:h, :] = score[-h:, :] = 0
    score[:, :h] = score[:, -h:] = 0
    if score.max() <= 0:
        return 64 - h, 64 - h
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return int(r - h), int(c - h)


def up_nearest(lr):
    return np.asarray(Image.fromarray(lr).resize((128, 128), Image.NEAREST))


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, path, bundle):
    fig.savefig(path)
    plt.close(fig)
    with open(path, "rb") as f:
        bundle[os.path.basename(path)] = base64.b64encode(f.read()).decode()
    print("saved", path)


def pick_examples(labels, ok_hr, classes, rng):
    out = []
    for c in classes:
        idx = np.flatnonzero((labels == c) & ok_hr)
        if len(idx) == 0:
            idx = np.flatnonzero(labels == c)
        out.append(int(rng.choice(idx)))
    return out


def fig_comparison(test, sr, methods, picks, labels, path, bundle):
    cols = ["LR 32 px"] + [LABEL.get(m, m) for m in methods] + ["Real HR 128 px"]
    n = len(picks)
    fig, axes = plt.subplots(2 * n, len(cols), figsize=(1.45 * len(cols), 2.9 * n))
    for r, i in enumerate(picks):
        imgs = [up_nearest(test["lr_bicubic"][i])] + [sr[m][i] for m in methods] + [test["hr"][i]]
        y0, x0 = zoom_box(test["hr"][i])
        for k, im in enumerate(imgs):
            a, z = axes[2 * r, k], axes[2 * r + 1, k]
            a.imshow(im, interpolation="nearest")
            a.add_patch(Rectangle((x0 - .5, y0 - .5), ZOOM, ZOOM, fill=False, ec=ACCENT, lw=1.2))
            z.imshow(im[y0:y0 + ZOOM, x0:x0 + ZOOM], interpolation="nearest")
            clean(a); clean(z)
            if r == 0:
                a.set_title(cols[k], fontsize=8.5)
        axes[2 * r, 0].set_ylabel(CLASS_SHORT[labels[i]], fontsize=8.5, color=INK)
        axes[2 * r + 1, 0].set_ylabel("zoom ×4", fontsize=8, color=INK_2)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, path, bundle)


def fig_error_maps(test, sr, methods, picks, labels, path, bundle):
    n = len(picks)
    fig, axes = plt.subplots(2 * n, len(methods) + 1, figsize=(1.6 * (len(methods) + 1), 3.2 * n))
    vmax_e, vmax_c = 0.15, 0.6
    for r, i in enumerate(picks):
        hr = test["hr"][i]
        g_hr = lum(hr)
        e_hr = sobel(g_hr)
        axes[2 * r, 0].imshow(hr); axes[2 * r, 0].set_ylabel(CLASS_SHORT[labels[i]], fontsize=8.5)
        axes[2 * r + 1, 0].imshow(e_hr, cmap="gray"); axes[2 * r + 1, 0].set_ylabel("HR contours", fontsize=8)
        for k, m in enumerate(methods, start=1):
            g = lum(sr[m][i])
            im1 = axes[2 * r, k].imshow(np.abs(g - g_hr), cmap="magma", vmin=0, vmax=vmax_e)
            im2 = axes[2 * r + 1, k].imshow(np.abs(sobel(g) - e_hr), cmap="magma", vmin=0, vmax=vmax_c)
            if r == 0:
                axes[0, k].set_title(LABEL.get(m, m), fontsize=8.5)
        if r == 0:
            axes[0, 0].set_title("Real HR", fontsize=8.5)
    for a in axes.flat:
        clean(a)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, right=0.9)
    c1 = fig.add_axes([0.915, 0.53, 0.012, 0.3]); fig.colorbar(im1, cax=c1).set_label("|SR − HR| luminance", fontsize=8)
    c2 = fig.add_axes([0.915, 0.15, 0.012, 0.3]); fig.colorbar(im2, cax=c2).set_label("|∇SR − ∇HR| (Sobel)", fontsize=8)
    save(fig, path, bundle)


def fig_failures(test, sr, best, pi_best, pi_hr, labels, path, bundle, k=3):
    hr_ok = pi_hr["pred_ens"] == labels
    flips = np.flatnonzero(hr_ok & (pi_best["pred_ens"] != labels))
    # most confident wrong answers first
    conf = pi_best["probs_ens"][flips, pi_best["pred_ens"][flips]]
    flips = flips[np.argsort(-conf)][:k]
    halluc = np.argsort(-pi_best["edge_err"])[:k]
    rows = [("prediction flipped", int(i)) for i in flips] + [("strongest contour error", int(i)) for i in halluc]
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.6, 1.75 * len(rows)))
    titles = ["LR 32 px", LABEL.get(best, best), "Real HR", "|SR − HR|"]
    for r, (kind, i) in enumerate(rows):
        hr = test["hr"][i]
        ims = [up_nearest(test["lr_bicubic"][i]), sr[best][i], hr]
        for c, im in enumerate(ims):
            axes[r, c].imshow(im, interpolation="nearest")
        axes[r, 3].imshow(np.abs(lum(sr[best][i]) - lum(hr)), cmap="magma", vmin=0, vmax=0.15)
        for c in range(4):
            clean(axes[r, c])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=8.5)
        axes[r, 0].set_ylabel(f"{kind}\ntrue: {CLASS_SHORT[labels[i]]}\nHR→{CLASS_SHORT[pi_hr['pred_ens'][i]]}"
                              f"\nSR→{CLASS_SHORT[pi_best['pred_ens'][i]]}", fontsize=7, rotation=0,
                              ha="right", va="center")
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    save(fig, path, bundle)


def fig_metrics(metrics, order, path, bundle):
    names = [m for m in order if m in metrics]
    f1 = np.array([metrics[m]["macro_f1_mean"] for m in names])
    sd = np.array([metrics[m]["macro_f1_std"] for m in names])
    psnr = np.array([metrics[m]["psnr_mean"] for m in names])
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(names))
    a.bar(x, f1, yerr=sd, color=BAR, width=0.62, capsize=3, error_kw={"ecolor": INK_2, "lw": 1})
    ref = metrics["real_HR"]["macro_f1_mean"]
    a.axhline(ref, color=INK_2, ls="--", lw=1)
    a.text(len(names) - 0.6, ref + 0.01, f"real HR {ref:.3f}", ha="right", fontsize=8, color=INK_2)
    for xi, v, e in zip(x, f1, sd):
        a.text(xi, v + e + 0.015, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    a.set_xticks(x); a.set_xticklabels([LABEL.get(m, m) for m in names], rotation=30, ha="right")
    a.set_ylabel("Macro F1 (frozen classifier, mean ± sd, 3 seeds)"); a.set_ylim(0, 0.9)
    a.yaxis.grid(True, color=GRID); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.set_title("Utility for classification", loc="left", fontsize=10)
    b.scatter(psnr, f1, s=40, color=BAR, edgecolor="white", linewidth=1.5, zorder=3)
    short = {"bicubic": "Bicubic", "A_vgg": "A+VGG", "B_vgg": "B+VGG", "C_vgg": "C+VGG"}
    rank = np.argsort(np.argsort(f1))  # alternate label side by F1 rank so clustered points stay readable
    for p, v, m, r in zip(psnr, f1, names, rank):
        b.annotate(short.get(m, m), (p, v), textcoords="offset points",
                   xytext=(6, 4) if r % 2 else (6, -11), fontsize=8, color=INK)
    b.set_xlabel("PSNR (dB) — pixel fidelity"); b.set_ylabel("Macro F1 — utility")
    b.grid(True, color=GRID); b.set_axisbelow(True)
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    b.set_title("Higher PSNR does not mean higher utility", loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, path, bundle)


def fig_per_class(metrics, order, path, bundle):
    names = [m for m in order if m in metrics] + ["real_HR"]
    mat = np.array([metrics[m]["per_class_recall_mean"] for m in names])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 1.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if mat[r, c] > 0.6 else INK)
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASS_SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels([LABEL.get(m, m) for m in names])
    ax.set_title("Per-class recall of the frozen classifier (mean of 3 seeds)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, path, bundle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--sr", default="outputs/sr_test")
    ap.add_argument("--eval", default="results/eval")
    ap.add_argument("--out", default="figures/results")
    ap.add_argument("--ref", default="A", help="reference SRGAN shown in the comparison")
    ap.add_argument("--improved", nargs="+", default=["B", "C"])
    ap.add_argument("--best", default="C", help="model analysed for failures")
    ap.add_argument("--order", nargs="+",
                    default=["bicubic", "A0", "A", "B", "C", "A_vgg", "B_vgg", "C_vgg"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    test = np.load(os.path.join(args.data, "test.npz"))
    labels = test["labels"].astype(int)
    metrics = json.load(open(os.path.join(args.eval, "metrics.json")))
    load_pi = lambda m: np.load(os.path.join(args.eval, "per_image", f"{m}.npz"))
    shown = ["bicubic", "A0", args.ref] + args.improved
    sr = {m: np.load(os.path.join(args.sr, f"{m}.npz"))["sr"] for m in set(shown + [args.best])}
    pi_hr = load_pi("real_HR")

    rng = np.random.default_rng(args.seed)
    ok = pi_hr["pred_ens"] == labels
    picks = pick_examples(labels, ok, [4, 5, 1, 3], rng)        # melanoma, nevus, BCC, dermatofibroma
    bundle = {}
    fig_comparison(test, sr, shown, picks, labels, os.path.join(args.out, "fig4_comparison_zoom.png"), bundle)
    fig_error_maps(test, sr, ["bicubic", "A0", args.ref, args.best], picks[:3], labels,
                   os.path.join(args.out, "fig5_error_maps.png"), bundle)
    fig_failures(test, sr, args.best, load_pi(args.best), pi_hr, labels,
                 os.path.join(args.out, "fig6_failures.png"), bundle)
    fig_metrics(metrics, args.order, os.path.join(args.out, "fig7_metrics.png"), bundle)
    fig_per_class(metrics, args.order, os.path.join(args.out, "fig8_per_class_recall.png"), bundle)

    summary = {"examples_shown_test_indices": picks}
    for m in metrics:
        if m == "real_HR" or not os.path.exists(os.path.join(args.eval, "per_image", f"{m}.npz")):
            continue
        p = load_pi(m)["pred_ens"]
        summary[m] = {"correct_to_wrong_vs_HR": int(np.sum(ok & (p != labels))),
                      "wrong_to_correct_vs_HR": int(np.sum(~ok & (p == labels))),
                      "n_test": int(len(labels))}
    with open(os.path.join(args.out, "failure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    bundle["failure_summary.json"] = base64.b64encode(json.dumps(summary, indent=2).encode()).decode()
    with open(os.path.join(args.out, "figures_b64.json"), "w") as f:
        json.dump(bundle, f)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
