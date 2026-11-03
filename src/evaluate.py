"""Evaluate super-resolution methods on the TEST split (run once, after all training is done).

For every method (bicubic baseline, each trained generator, and the real HR images as the
reference ceiling) it computes:
  image quality      PSNR, SSIM, LPIPS (AlexNet), edge error (L1 of Sobel magnitude)
  lesion stability   Dice between Otsu lesion masks of SR and HR; classifier agreement with HR
  application        frozen ResNet-18: macro F1 and per-class recall, mean ± std over 3 seeds
  uncertainty        bootstrap 95% CI over test images (1000 resamples)
  cost               generator parameters, inference ms / image
and saves the super-resolved test images (deliverable) plus results tables.

Usage (Kaggle):
    python src/evaluate.py --runs runs/sr/A0 runs/sr/A runs/sr/B runs/sr/C \
        --classifier runs/classifier --out results/eval
Each run folder must contain config.json and G_final.pt (written by train_sr.py).
Degraded-input runs (lr_key = lr_degraded) are compared to bicubic on the degraded input.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score, recall_score

from losses import psnr, sobel_magnitude, ssim
from models import Generator, count_params
from train_classifier import build_model, predict

NUM_CLASSES = 7
N_BOOT = 1000


def to01(x_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div_(255.0)


def to_uint8(x01: torch.Tensor) -> np.ndarray:
    return (x01.clamp(0, 1).mul(255).add(0.5).byte().permute(0, 2, 3, 1).cpu().numpy())


@torch.no_grad()
def run_generator(run_dir, lr_uint8, device, batch_size=64):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    G = Generator(n_blocks=cfg["n_blocks"], attention=cfg["attention"]).to(device).eval()
    G.load_state_dict(torch.load(os.path.join(run_dir, "G_final.pt"), map_location=device))
    lr = to01(lr_uint8) * 2 - 1
    outs = []
    if device.type == "cuda":  # warm-up so timing excludes CUDA initialisation
        G(lr[:batch_size].to(device)); torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, len(lr), batch_size):
        sr = G(lr[i:i + batch_size].to(device))
        outs.append(((sr.float() + 1) / 2).clamp(0, 1).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / len(lr)
    return torch.cat(outs), {"params_G": count_params(G), "ms_per_image": round(ms, 3),
                             "lr_key": cfg["lr_key"], "config": cfg}


def bicubic(lr_uint8):
    return F.interpolate(to01(lr_uint8), scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)


def lesion_masks(x01: torch.Tensor) -> np.ndarray:
    """Otsu threshold on luminance; lesion = darker side. Returns bool (N,H,W)."""
    lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).numpy()
    masks = np.zeros(lum.shape, dtype=bool)
    for i, g in enumerate(lum):
        masks[i] = g < threshold_otsu(g) if g.max() > g.min() else False
    return masks


def dice(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = (a & b).reshape(len(a), -1).sum(1)
    tot = a.reshape(len(a), -1).sum(1) + b.reshape(len(b), -1).sum(1)
    return np.where(tot > 0, 2 * inter / np.maximum(tot, 1), 1.0)


def per_image_quality(sr, hr, lpips_fn, device, batch_size=128):
    out = {"psnr": [], "ssim": [], "lpips": [], "edge_err": []}
    with torch.no_grad():
        for i in range(0, len(sr), batch_size):
            x, y = sr[i:i + batch_size].to(device), hr[i:i + batch_size].to(device)
            out["psnr"].append(psnr(x, y, reduce=False).cpu())
            out["ssim"].append(ssim(x, y, reduce=False).cpu())
            out["edge_err"].append((sobel_magnitude(x) - sobel_magnitude(y)).abs().flatten(1).mean(1).cpu())
            if lpips_fn is not None:
                out["lpips"].append(lpips_fn(x * 2 - 1, y * 2 - 1).flatten().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def bootstrap_ci(values=None, y=None, pred=None, seed=0):
    """95% percentile CI of the mean of `values`, or of macro F1 for (y, pred)."""
    rng = np.random.default_rng(seed)
    n = len(values) if values is not None else len(y)
    stats = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stats.append(values[idx].mean() if values is not None else macro_f1(y[idx], pred[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs", nargs="*", default=[], help="train_sr.py run folders")
    ap.add_argument("--classifier", default="runs/classifier", help="folder with seed*/best.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--save-images", default="outputs/sr_test", help="'' to skip saving SR images")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: N test images")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    d = np.load(os.path.join(args.data, "test.npz"))
    sl = slice(0, args.limit or None)
    hr, y = to01(d["hr"][sl]), d["labels"][sl].astype(np.int64)
    lr = {"lr_bicubic": d["lr_bicubic"][sl], "lr_degraded": d["lr_degraded"][sl]}

    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    classifiers = []
    for s in args.seeds:
        m = build_model(pretrained=False).to(device)
        m.load_state_dict(torch.load(os.path.join(args.classifier, f"seed{s}", "best.pt"), map_location=device))
        classifiers.append(m.eval())

    # ---- methods: real HR (ceiling), bicubic on both inputs, every trained run
    methods = [("real_HR", "-", hr, {"params_G": 0, "ms_per_image": 0.0})]
    for key, label in (("lr_bicubic", "bicubic"), ("lr_degraded", "bicubic_degraded")):
        t0 = time.time()
        img = bicubic(lr[key])
        methods.append((label, key, img, {"params_G": 0, "ms_per_image": round((time.time() - t0) * 1000 / len(img), 3)}))
    for run in args.runs:
        cfg = json.load(open(os.path.join(run, "config.json")))
        img, info = run_generator(run, lr[cfg["lr_key"]], device)
        methods.append((cfg["name"], cfg["lr_key"], img, info))

    hr_masks = lesion_masks(hr)
    hr_preds = [predict(m, hr, device).argmax(1) for m in classifiers]

    rows, per_class_rows, full = [], [], {}
    for name, key, img, info in methods:
        q = per_image_quality(img, hr, lpips_fn, device)
        q["dice"] = dice(lesion_masks(img), hr_masks)
        probs = [predict(m, img, device) for m in classifiers]
        preds = [p.argmax(1) for p in probs]
        f1s = np.array([macro_f1(y, p) for p in preds])
        recalls = np.array([recall_score(y, p, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)
                            for p in preds])
        agree = np.array([(p == h).mean() for p, h in zip(preds, hr_preds)])
        ens_pred = np.mean(probs, axis=0).argmax(1)

        row = {"method": name, "input": key,
               "macro_f1_mean": f1s.mean(), "macro_f1_std": f1s.std(ddof=1) if len(f1s) > 1 else 0.0,
               "macro_f1_ci95": bootstrap_ci(y=y, pred=ens_pred),
               "agreement_with_HR_mean": agree.mean(),
               "params_G": info["params_G"], "ms_per_image": info["ms_per_image"]}
        for k, v in q.items():
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_ci95"] = bootstrap_ci(values=v)
        rows.append(row)
        per_class_rows.append({"method": name, **{f"recall_c{c}": float(recalls[:, c].mean()) for c in range(NUM_CLASSES)}})
        full[name] = {**row, "per_seed_macro_f1": f1s.tolist(), "per_class_recall_mean": recalls.mean(0).tolist(),
                      "config": info.get("config")}
        print(f"{name:22s} PSNR {row['psnr_mean']:6.2f}  SSIM {row['ssim_mean']:.4f}  "
              + (f"LPIPS {row['lpips_mean']:.4f}  " if "lpips_mean" in row else "")
              + f"Dice {row['dice_mean']:.4f}  F1 {row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}  "
              f"agree {row['agreement_with_HR_mean']:.4f}", flush=True)

        if args.save_images and name != "real_HR":
            os.makedirs(args.save_images, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_images, f"{name}.npz"), sr=to_uint8(img), labels=y,
                                source="DermaMNIST 128 test split (CC BY-NC 4.0); synthetic, not for diagnosis",
                                method=name, input=key)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(full, f, indent=2, default=float)
    cols = ["method", "input", "psnr_mean", "ssim_mean", "lpips_mean", "edge_err_mean", "dice_mean",
            "macro_f1_mean", "macro_f1_std", "agreement_with_HR_mean", "params_G", "ms_per_image"]
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "") for c in cols])
    with open(os.path.join(args.out, "per_class_recall.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class_rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(per_class_rows)
    print("saved", args.out)


if __name__ == "__main__":
    main()
