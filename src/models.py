"""SRGAN generator / discriminator for x4 super-resolution (32 -> 128 px).

Generator: SRResNet (Ledig et al., 2017) with optional channel attention (variant B/C).
Discriminator: SRGAN-style convolutional discriminator.
Both are trained from scratch: no pretrained weights in the generator (Fiche 07).

Structure follows the reference implementation https://github.com/leftthomas/SRGAN
(starter code authorised by the project), rewritten and extended for this project.
"""
import math

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention (variant B): re-weights feature channels
    using a global descriptor of the whole patch."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, attention: bool):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.attn = ChannelAttention(channels) if attention else nn.Identity()

    def forward(self, x):
        return x + self.attn(self.body(x))


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, 3, padding=1),
            nn.PixelShuffle(factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.body(x)


class Generator(nn.Module):
    """Input: LR image in [-1, 1], (N,3,32,32). Output: SR image in [-1, 1], (N,3,128,128)."""

    def __init__(self, scale: int = 4, n_blocks: int = 16, channels: int = 64, attention: bool = False):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, channels, 9, padding=4), nn.PReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels, attention) for _ in range(n_blocks)])
        self.trunk_out = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels))
        self.upsample = nn.Sequential(*[UpsampleBlock(channels, 2) for _ in range(int(math.log2(scale)))])
        self.tail = nn.Conv2d(channels, 3, 9, padding=4)

    def forward(self, x):
        h = self.head(x)
        h = h + self.trunk_out(self.blocks(h))
        return torch.tanh(self.tail(self.upsample(h)))


class Discriminator(nn.Module):
    """SRGAN discriminator. Input: HR or SR image in [-1, 1], (N,3,128,128). Output: logits (N,)."""

    def __init__(self, base: int = 64):
        super().__init__()

        def block(cin, cout, stride, bn=True):
            layers = [nn.Conv2d(cin, cout, 3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        c = base
        self.features = nn.Sequential(
            *block(3, c, 1, bn=False), *block(c, c, 2),
            *block(c, 2 * c, 1), *block(2 * c, 2 * c, 2),
            *block(2 * c, 4 * c, 1), *block(4 * c, 4 * c, 2),
            *block(4 * c, 8 * c, 1), *block(8 * c, 8 * c, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(8 * c, 16 * c, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16 * c, 1, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).flatten()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
