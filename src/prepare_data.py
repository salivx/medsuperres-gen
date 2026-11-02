"""Step 1 of MedSuperRes Gen: download DermaMNIST (128 px), freeze the official
splits, build the low-resolution inputs and count classes.

Usage (Kaggle / Colab / local):
    python src/prepare_data.py --out data

Outputs
    data/processed/{split}.npz      hr (N,128,128,3) uint8, lr_bicubic (N,32,32,3),
                                    lr_degraded (N,32,32,3), labels (N,)
    splits/{split}.csv              frozen file list: index, label, sha1 of the HR image
    splits/class_counts.csv         per-class counts per split (for the data card)
    splits/degradation.json         exact degradation parameters (for reproducibility)

The splits are the official MedMNIST+ ones and are never reshuffled.
The test split is never used to train or tune anything.
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageFilter

SCALE = 4
HR_SIZE = 128
LR_SIZE = HR_SIZE // SCALE  # 32

# Second degradation: bicubic + blur + controlled noise (Fiche 07, "robustesse").
BLUR_SIGMA = 1.0          # Gaussian blur on the HR image before downsampling
NOISE_STD = 5.0 / 255.0   # additive Gaussian noise on the LR image, in [0,1] units
NOISE_SEED = 2026         # fixed so the degraded inputs are identical for everyone

SPLITS = ("train", "val", "test")


def bicubic_down(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((LR_SIZE, LR_SIZE), Image.BICUBIC))


def degraded_down(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    blurred = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
    lr = np.asarray(blurred.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)).astype(np.float32) / 255.0
    lr = lr + rng.normal(0.0, NOISE_STD, size=lr.shape).astype(np.float32)
    return (np.clip(lr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_dermamnist(root: str):
    from medmnist import DermaMNIST, INFO

    os.makedirs(root, exist_ok=True)
    data = {}
    for split in SPLITS:
        ds = DermaMNIST(split=split, download=True, size=HR_SIZE, root=root)
        imgs = ds.imgs
        if imgs.ndim == 3:  # safety: grayscale -> RGB
            imgs = np.repeat(imgs[..., None], 3, axis=-1)
        data[split] = (imgs.astype(np.uint8), ds.labels.reshape(-1).astype(np.int64))
    return data, INFO["dermamnist"]["label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="folder for raw + processed data")
    ap.add_argument("--splits-dir", default="splits", help="folder for frozen split lists")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "raw")
    proc_dir = os.path.join(args.out, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)

    data, label_names = load_dermamnist(raw_dir)
    class_ids = sorted(int(k) for k in label_names)
    counts = {split: np.bincount(data[split][1], minlength=len(class_ids)) for split in SPLITS}

    rng = np.random.default_rng(NOISE_SEED)
    for split in SPLITS:
        hr, labels = data[split]
        assert hr.shape[1:] == (HR_SIZE, HR_SIZE, 3), hr.shape

        lr_bic = np.stack([bicubic_down(x) for x in hr])
        lr_deg = np.stack([degraded_down(x, rng) for x in hr])
        np.savez_compressed(os.path.join(proc_dir, f"{split}.npz"),
                            hr=hr, lr_bicubic=lr_bic, lr_degraded=lr_deg, labels=labels)

        with open(os.path.join(args.splits_dir, f"{split}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "label", "sha1_hr"])
            for i, (img, y) in enumerate(zip(hr, labels)):
                w.writerow([i, int(y), hashlib.sha1(img.tobytes()).hexdigest()])
        print(f"{split:5s}: {len(labels):5d} images  HR {hr.shape[1:]}  LR {lr_bic.shape[1:]}")

    with open(os.path.join(args.splits_dir, "class_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", *SPLITS, "train_pct"])
        n_train = counts["train"].sum()
        for c in class_ids:
            w.writerow([c, label_names[str(c)], *(int(counts[s][c]) for s in SPLITS),
                        f"{100 * counts['train'][c] / n_train:.1f}"])

    with open(os.path.join(args.splits_dir, "degradation.json"), "w") as f:
        json.dump({"scale": SCALE, "hr_size": HR_SIZE, "lr_size": LR_SIZE,
                   "bicubic": "PIL Image.BICUBIC",
                   "degraded": {"blur": "PIL GaussianBlur on HR", "blur_sigma": BLUR_SIGMA,
                                "noise": "additive Gaussian on LR", "noise_std_0_1": NOISE_STD,
                                "noise_seed": NOISE_SEED, "order": "blur -> bicubic -> noise"}},
                  f, indent=2)

    print("\nClass counts (train / val / test):")
    for c in class_ids:
        print(f"  {c} {label_names[str(c)][:40]:40s} "
              + " / ".join(str(int(counts[s][c])) for s in SPLITS))


if __name__ == "__main__":
    main()
