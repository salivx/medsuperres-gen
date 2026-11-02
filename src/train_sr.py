"""Train the x4 super-resolution models of MedSuperRes Gen (Fiche 07).

Variants (same data, same iteration budget, same seed):
    A0  SRResNet, L1 pixel loss only                      -> isolates the adversarial effect
    A   SRGAN reference: L1 + adversarial
    B   A + channel attention in every residual block
    C   B + structure-preservation loss (Sobel edges + (1 - SSIM))
Options:
    --vgg         add the frozen VGG19 perceptual loss (declared; ablate by running with/without)
    --lr-key      lr_bicubic (main) or lr_degraded (robustness: blur + noise)

GAN variants start with an L1-only warm-up (--warmup-frac of the budget) inside the SAME total
budget as A0, so all variants see exactly the same number of generator updates.

Kaggle sessions get interrupted: a checkpoint is written every epoch and --resume continues it.

Usage:
    python src/train_sr.py --variant A0 --out runs/sr
    python src/train_sr.py --variant C --vgg --out runs/sr
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from losses import VGGPerceptual, edge_loss, psnr, ssim
from models import Discriminator, Generator, count_params

PRESETS = {
    "A0": dict(gan=False, attention=False, struct=False),
    "A": dict(gan=True, attention=False, struct=False),
    "B": dict(gan=True, attention=True, struct=False),
    "C": dict(gan=True, attention=True, struct=True),
}


def to_pm1(x_uint8: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [-1, 1]."""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)


def to01(x: torch.Tensor) -> torch.Tensor:
    return (x.float() + 1.0) / 2.0


def paired_augment(lr: torch.Tensor, hr: torch.Tensor):
    """Same random flip / 90° rotation on the LR input and its HR target."""
    if torch.rand(1).item() < 0.5:
        lr, hr = lr.flip(3), hr.flip(3)
    if torch.rand(1).item() < 0.5:
        lr, hr = lr.flip(2), hr.flip(2)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        lr, hr = torch.rot90(lr, k, (2, 3)), torch.rot90(hr, k, (2, 3))
    return lr, hr


@torch.no_grad()
def evaluate(G, lr, hr, device, batch_size=128):
    G.eval()
    p_sr, s_sr, p_bic, s_bic = [], [], [], []
    for i in range(0, len(lr), batch_size):
        x = lr[i:i + batch_size].to(device)
        y = to01(hr[i:i + batch_size].to(device))
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            sr = G(x)
        sr = to01(sr).clamp(0, 1)
        bic = to01(F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)).clamp(0, 1)
        p_sr.append(psnr(sr, y, reduce=False)); s_sr.append(ssim(sr, y, reduce=False))
        p_bic.append(psnr(bic, y, reduce=False)); s_bic.append(ssim(bic, y, reduce=False))
    G.train()
    cat = lambda t: float(torch.cat(t).mean())
    return {"psnr": cat(p_sr), "ssim": cat(s_sr), "psnr_bicubic": cat(p_bic), "ssim_bicubic": cat(s_bic)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(PRESETS), required=True)
    ap.add_argument("--vgg", action="store_true", help="add frozen VGG19 perceptual loss")
    ap.add_argument("--lr-key", default="lr_bicubic", choices=["lr_bicubic", "lr_degraded"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/sr")
    ap.add_argument("--name", default=None, help="run name (default: built from the options)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--warmup-frac", type=float, default=0.25, help="L1-only fraction for GAN variants")
    ap.add_argument("--w-adv", type=float, default=1e-3)
    ap.add_argument("--w-vgg", type=float, default=6e-3)
    ap.add_argument("--w-edge", type=float, default=0.1)
    ap.add_argument("--w-ssim", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: N images per split")
    ap.add_argument("--debug-random-vgg", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.variant])
    name = args.name or "_".join(
        [args.variant] + (["vgg"] if args.vgg else []) + (["degraded"] if args.lr_key == "lr_degraded" else []))
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    def load(split):
        d = np.load(os.path.join(args.data, f"{split}.npz"))
        lr, hr = d[args.lr_key], d["hr"]
        if args.limit:
            lr, hr = lr[:args.limit], hr[:args.limit]
        return to_pm1(lr), to_pm1(hr)

    tr_lr, tr_hr = load("train")
    va_lr, va_hr = load("val")  # the test split is never loaded during training
    loader = DataLoader(TensorDataset(tr_lr, tr_hr), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=2, pin_memory=amp,
                        generator=torch.Generator().manual_seed(args.seed))
    total_iters = args.epochs * len(loader)
    warmup_iters = int(args.warmup_frac * total_iters) if cfg["gan"] else total_iters

    G = Generator(n_blocks=args.n_blocks, attention=cfg["attention"]).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[int(0.75 * total_iters)], gamma=0.1)
    scaler_g = torch.amp.GradScaler("cuda", enabled=amp)
    D = opt_d = scaler_d = sched_d = None
    if cfg["gan"]:
        D = Discriminator().to(device)
        opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.9, 0.999))
        sched_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[int(0.75 * total_iters)], gamma=0.1)
        scaler_d = torch.amp.GradScaler("cuda", enabled=amp)
    vgg = VGGPerceptual(pretrained=not args.debug_random_vgg).to(device) if args.vgg else None

    start_epoch, it, history, best_psnr, gpu_seconds = 1, 0, [], -1.0, 0.0
    ckpt_path = os.path.join(out, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(ck["G"]); opt_g.load_state_dict(ck["opt_g"]); sched_g.load_state_dict(ck["sched_g"])
        scaler_g.load_state_dict(ck["scaler_g"])
        if D is not None:
            D.load_state_dict(ck["D"]); opt_d.load_state_dict(ck["opt_d"]); sched_d.load_state_dict(ck["sched_d"])
            scaler_d.load_state_dict(ck["scaler_d"])
        start_epoch, it, history = ck["epoch"] + 1, ck["iter"], ck["history"]
        best_psnr, gpu_seconds = ck["best_psnr"], ck["gpu_seconds"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed from epoch {ck['epoch']}")

    config = {**vars(args), **cfg, "name": name, "total_iters": total_iters, "warmup_iters": warmup_iters,
              "params_G": count_params(G), "params_D": count_params(D) if D else 0, "device": str(device)}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({k: config[k] for k in ("name", "gan", "attention", "struct", "vgg", "lr_key",
                                             "total_iters", "warmup_iters", "params_G")}))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        sums = {}
        for lr_b, hr_b in loader:
            lr_b, hr_b = paired_augment(lr_b.to(device, non_blocking=True), hr_b.to(device, non_blocking=True))
            adversarial = cfg["gan"] and it >= warmup_iters

            # ---- discriminator
            if adversarial:
                with torch.autocast(device_type=device.type, enabled=amp):
                    with torch.no_grad():
                        fake = G(lr_b)
                    d_real, d_fake = D(hr_b), D(fake)
                    loss_d = (F.binary_cross_entropy_with_logits(d_real.float(), torch.ones_like(d_real.float()))
                              + F.binary_cross_entropy_with_logits(d_fake.float(), torch.zeros_like(d_fake.float())))
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d); scaler_d.update()
                sums["loss_d"] = sums.get("loss_d", 0.0) + loss_d.item()

            # ---- generator
            with torch.autocast(device_type=device.type, enabled=amp):
                sr = G(lr_b)
            sr01, hr01 = to01(sr), to01(hr_b)
            terms = {"l1": F.l1_loss(sr01, hr01)}
            if adversarial:
                with torch.autocast(device_type=device.type, enabled=amp):
                    d_sr = D(sr).float()
                terms["adv"] = args.w_adv * F.binary_cross_entropy_with_logits(d_sr, torch.ones_like(d_sr))
                if vgg is not None:
                    terms["vgg"] = args.w_vgg * vgg(sr01, hr01)
            if cfg["struct"]:
                terms["edge"] = args.w_edge * edge_loss(sr01, hr01)
                terms["ssim"] = args.w_ssim * (1 - ssim(sr01, hr01))
            loss_g = sum(terms.values())
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g); scaler_g.update()
            sched_g.step()
            if sched_d is not None:
                sched_d.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            it += 1

        gpu_seconds += time.time() - t0
        val = evaluate(G, va_lr, va_hr, device)
        row = {"epoch": epoch, "iter": it, **{k: v / len(loader) for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"[{name}] epoch {epoch:3d}/{args.epochs}  " + "  ".join(
            f"{k} {v:.4f}" for k, v in row.items() if k not in ("epoch", "iter")), flush=True)

        if val["psnr"] > best_psnr:
            best_psnr = val["psnr"]
            torch.save(G.state_dict(), os.path.join(out, "G_best_val_psnr.pt"))
        ck = {"G": G.state_dict(), "opt_g": opt_g.state_dict(), "sched_g": sched_g.state_dict(),
              "scaler_g": scaler_g.state_dict(), "epoch": epoch, "iter": it, "history": history,
              "best_psnr": best_psnr, "gpu_seconds": gpu_seconds, "rng": torch.get_rng_state()}
        if D is not None:
            ck.update(D=D.state_dict(), opt_d=opt_d.state_dict(), sched_d=sched_d.state_dict(),
                      scaler_d=scaler_d.state_dict())
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic: an interruption never leaves a broken file

    torch.save(G.state_dict(), os.path.join(out, "G_final.pt"))
    summary = {"name": name, "final_val": history[-1], "best_val_psnr": best_psnr,
               "gpu_minutes": round(gpu_seconds / 60, 1), "history": history}
    with open(os.path.join(out, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done: {name}  val PSNR {history[-1]['val_psnr']:.2f} dB (bicubic "
          f"{history[-1]['val_psnr_bicubic']:.2f})  SSIM {history[-1]['val_ssim']:.4f}  "
          f"GPU {summary['gpu_minutes']} min")


if __name__ == "__main__":
    main()
