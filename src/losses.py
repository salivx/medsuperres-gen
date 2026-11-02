"""Losses and image-quality metrics shared by training and evaluation.

All functions take images in [0, 1], shape (N,3,H,W), unless stated otherwise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- SSIM / PSNR


def _gaussian_window(size: int = 11, sigma: float = 1.5, channels: int = 3, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()
    return w


def ssim(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """Standard SSIM (Wang et al., 2004): 11x11 Gaussian window, sigma 1.5, data range 1.
    Computed per channel, averaged. Returns a scalar, or one value per image if reduce=False."""
    x, y = x.float(), y.float()
    c = x.shape[1]
    w = _gaussian_window(channels=c, device=x.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    per_image = m.flatten(1).mean(1)
    return per_image.mean() if reduce else per_image


def psnr(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    mse = (x.float() - y.float()).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    per_image = 10 * torch.log10(1.0 / mse)
    return per_image.mean() if reduce else per_image


# ---------------------------------------------------------------- edges


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of the luminance image, (N,1,H,W)."""
    lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).float()
    kx = _SOBEL_X.to(x.device).view(1, 1, 3, 3)
    ky = _SOBEL_X.t().to(x.device).view(1, 1, 3, 3)
    gx = F.conv2d(lum, kx, padding=1)
    gy = F.conv2d(lum, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def edge_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """L1 between Sobel gradient magnitudes (variant C, structure preservation)."""
    return F.l1_loss(sobel_magnitude(sr), sobel_magnitude(hr))


# ---------------------------------------------------------------- perceptual (frozen VGG)


class VGGPerceptual(nn.Module):
    """Frozen VGG19 feature loss (Ledig et al., 2017). Allowed by the project rules as a frozen
    loss network: declared here and ablated with --no-vgg. Never part of the generator, and never
    used for evaluation (the evaluator is the separate ResNet-18 classifier)."""

    def __init__(self, layer: int = 35, pretrained: bool = True):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        # features[:36] ends at relu5_4, the layer used for the SRGAN "VGG54" loss.
        self.features = vgg.features[: layer + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        f = lambda x: self.features((x.float() - self.mean) / self.std)
        return F.mse_loss(f(sr), f(hr))
