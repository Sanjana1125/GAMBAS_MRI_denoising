from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .path_generate import gilbert2d, generate_gilbert_indices_2D
from .pure_mamba import HilbertBiGRU, PureMamba


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=True),
            norm_layer(dim, affine=False),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=True),
            norm_layer(dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def build_hilbert_permutation(height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return serialization and inverse indices, both shaped [L]."""
    generator = gilbert2d(width, height)  # library expects W, H
    index_tensor = generate_gilbert_indices_2D(height, width, generator).reshape(-1).long()
    expected = height * width
    if index_tensor.numel() != expected:
        raise RuntimeError(f"Hilbert path length {index_tensor.numel()} != {expected}")
    if index_tensor.min().item() != 0 or index_tensor.max().item() != expected - 1:
        raise RuntimeError("Hilbert indices do not cover the expected raster range")
    if torch.unique(index_tensor).numel() != expected:
        raise RuntimeError("Hilbert path contains duplicate or missing raster indices")
    inverse = torch.argsort(index_tensor, dim=0)
    return index_tensor.to(device), inverse.to(device)


class HilbertSequenceLayer(nn.Module):
    """Serialize 2D features along a Hilbert path, mix, and restore raster order."""

    def __init__(self, dim: int, backend: str = "gru"):
        super().__init__()
        self.dim = dim
        self.backend = backend
        if backend == "gru":
            self.sequence_model = HilbertBiGRU(dim)
        elif backend == "pure_mamba":
            self.sequence_model = PureMamba(dim)
        elif backend == "identity":
            self.sequence_model = nn.Identity()
        else:
            raise ValueError(f"Unsupported sequence backend: {backend}")
        self._cache: Dict[Tuple[int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _indices(self, h: int, w: int, device: torch.device):
        key = (h, w, str(device))
        if key not in self._cache:
            self._cache[key] = build_hilbert_permutation(h, w, device)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {c}")
        permutation, inverse = self._indices(h, w, x.device)
        raster = x.flatten(2).transpose(1, 2).contiguous()  # [B,L,C]
        sequence = raster.index_select(1, permutation)
        mixed = self.sequence_model(sequence)
        restored = mixed.index_select(1, inverse)
        return restored.transpose(1, 2).contiguous().reshape(b, c, h, w)


class SequenceResidualBlock(nn.Module):
    """CNN-dominant block with a gently gated Hilbert sequence branch."""

    def __init__(self, channels: int = 256, backend: str = "gru"):
        super().__init__()
        self.cnn = ResnetBlock(channels)
        self.sequence = HilbertSequenceLayer(channels, backend=backend)
        self.sequence_scale = nn.Parameter(torch.tensor(-2.0))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn(x)
        seq = self.sequence(cnn)
        seq_delta = seq - cnn
        gated_seq = cnn + torch.sigmoid(self.sequence_scale) * seq_delta
        return self.fuse(torch.cat([cnn, gated_seq], dim=1)) + x


class ResNet2DGenerator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, ngf=64, n_blocks=9,
                 sequence_backend: str | None = None,
                 sequence_positions=(0, 4, 8)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReplicationPad2d(3),
            nn.Conv2d(input_dim, ngf, 7),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4, affine=False),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for i in range(n_blocks):
            if sequence_backend is not None and i in sequence_positions:
                blocks.append(SequenceResidualBlock(ngf * 4, backend=sequence_backend))
            else:
                blocks.append(ResnetBlock(ngf * 4))
        self.bottleneck = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2, affine=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf, affine=False),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(3),
            nn.Conv2d(ngf, output_dim, 7),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_h, original_w = x.shape[-2:]
        # Pad only when necessary so two down/up-sampling stages are invertible.
        pad_h = (4 - original_h % 4) % 4
        pad_w = (4 - original_w % 4) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.decoder(self.bottleneck(self.encoder(x)))
        return y[..., :original_h, :original_w]


class GAMBAS(ResNet2DGenerator):
    """Compatibility name: stable 2D Hilbert-BiGRU GAMBAS adaptation."""
    def __init__(self, input_dim=1, img_size=None, output_dim=1, backend="gru"):
        super().__init__(input_dim=input_dim, output_dim=output_dim,
                         sequence_backend=backend)


def test_hilbert_roundtrip(height=34, width=38, channels=4):
    device = torch.device("cpu")
    p, inv = build_hilbert_permutation(height, width, device)
    x = torch.arange(height * width * channels).reshape(1, height * width, channels)
    y = x.index_select(1, p).index_select(1, inv)
    assert torch.equal(x, y), "Hilbert round-trip failed"
    return True
