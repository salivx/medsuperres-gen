"""Frozen evaluation classifier for MedSuperRes Gen.

A ResNet-18 trained on the REAL 128 px training images only. Once trained it is frozen and applied,
unchanged, to the bicubic / A0 / A / B / C / real versions of the test set.

Rules followed (project PDF, "Réseaux préentraînés autorisés"):
  * ImageNet weights are allowed because this network is used ONLY for evaluation.
  * It is never used inside a generator loss (that would bias the comparison).
  * Checkpoint selection uses validation macro F1; the test set is only scored at the end.

Usage (Kaggle GPU):
    python src/train_classifier.py --data data/processed --out runs/classifier --seeds 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 7
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [0,1]."""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_model(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def augment(x: torch.Tensor) -> torch.Tensor:
    """Label-preserving dermoscopy augmentations, applied on the GPU batch.
    Random flips + 90° rotations (lesions have no canonical orientation)."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1).item() < 0.5:
        x = x.flip(2)
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(x, k, dims=(2, 3)) if k else x


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor, device, batch_size: int = 256) -> np.ndarray:
    """Return softmax probabilities (N,7) for images in [0,1], shape (N,3,128,128).
    This is the function every later evaluation (bicubic, A0, A, B, C) calls."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        xb = images[i:i + batch_size].to(device, non_blocking=True)
        if xb.shape[-1] != 128:
            raise ValueError(f"classifier expects 128 px inputs, got {tuple(xb.shape)}")
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(normalize(xb))
        out.append(F.softmax(logits.float(), dim=1).cpu())
    return torch.cat(out).numpy()


def scores(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(1)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": [float(r) for r in recall_score(
            y_true, y_pred, average=None, labels=list(range(NUM_CLASSES)), zero_division=0)],
    }


def train_one_seed(seed, tr, va, te, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = os.path.join(args.out, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    x_tr, y_tr = tr
    # Class-weighted loss (inverse frequency, normalized) against the 67% nevus imbalance.
    counts = np.bincount(y_tr.numpy(), minlength=NUM_CLASSES)
    w = torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32, device=device)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, generator=g, num_workers=2, pin_memory=device.type == "cuda")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1, best_epoch, history = -1.0, -1, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = augment(xb.to(device, non_blocking=True))
            yb = yb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.cross_entropy(model(normalize(xb)), yb, weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()

        val = scores(va[1].numpy(), predict(model, va[0], device))
        history.append({"epoch": epoch, "train_loss": running / len(loader),
                        "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
        print(f"[seed {seed}] epoch {epoch:3d}  loss {running / len(loader):.4f}  "
              f"val F1 {val['macro_f1']:.4f}  acc {val['accuracy']:.4f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch = val["macro_f1"], epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    # Test is scored ONCE, with the checkpoint chosen on validation.
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    test_probs = predict(model, te[0], device)
    np.save(os.path.join(out_dir, "test_probs_real_hr.npy"), test_probs)
    result = {
        "seed": seed, "best_epoch": best_epoch, "val_macro_f1": best_f1,
        "test_real_hr": scores(te[1].numpy(), test_probs),
        "gpu_minutes": round((time.time() - t0) / 60, 1), "device": str(device),
        "config": vars(args), "history": history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/classifier")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="train from scratch (ablation)")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    def load(split):
        d = np.load(os.path.join(args.data, f"{split}.npz"))
        x, y = d["hr"], d["labels"]
        if args.limit:
            x, y = x[:args.limit], y[:args.limit]
        return to_tensor(x), torch.from_numpy(y.astype(np.int64))

    tr, va, te = load("train"), load("val"), load("test")
    os.makedirs(args.out, exist_ok=True)

    results = [train_one_seed(s, tr, va, te, args, device) for s in args.seeds]

    f1s = np.array([r["test_real_hr"]["macro_f1"] for r in results])
    rec = np.array([r["test_real_hr"]["per_class_recall"] for r in results])
    summary = {
        "seeds": args.seeds,
        "test_real_hr_macro_f1_mean": float(f1s.mean()),
        "test_real_hr_macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        "test_real_hr_per_class_recall_mean": rec.mean(0).round(4).tolist(),
        "best_epochs": [r["best_epoch"] for r in results],
        "gpu_minutes_total": round(sum(r["gpu_minutes"] for r in results), 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
