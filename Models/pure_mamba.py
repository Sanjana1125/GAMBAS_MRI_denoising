"""Native-PyTorch sequence alternatives.

No mamba-ssm installation is required. The recommended backend for this small
2D MRI dataset is HilbertBiGRU. PureMamba is retained only as an experimental
Mamba-inspired recurrent selective scan.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HilbertBiGRU(nn.Module):
    """Stable bidirectional sequence mixer for tensors shaped [B, L, C]."""

    def __init__(self, d_model: int, hidden_size: int | None = None):
        super().__init__()
        hidden_size = hidden_size or max(32, d_model // 2)
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_size * 2, d_model)
        self.gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid ~= 0.12

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y, _ = self.gru(self.norm(x))
        y = self.proj(y)
        return residual + torch.sigmoid(self.gate) * y


class PureMamba(nn.Module):
    """Experimental Mamba-inspired block implemented only with PyTorch.

    It is deliberately recurrent rather than based on cumulative products,
    avoiding the severe underflow present in the uploaded implementation.
    Input/output: [B, L, d_model].
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.dwconv = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )
        self.param_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        u, z = self.in_proj(x).chunk(2, dim=-1)
        length = u.shape[1]
        u = self.dwconv(u.transpose(1, 2))[..., :length].transpose(1, 2)
        u = F.silu(u)

        params = self.param_proj(u)
        delta_raw, b_param, c_param = torch.split(
            params, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(delta_raw).clamp(max=1.0) + 1e-4
        A = -torch.exp(self.A_log).to(dtype=u.dtype, device=u.device)
        state = torch.zeros(
            u.shape[0], self.d_inner, self.d_state,
            dtype=u.dtype, device=u.device
        )
        outputs = []
        for t in range(length):
            dt = delta[:, t].unsqueeze(-1)
            transition = torch.exp(dt * A.unsqueeze(0))
            drive = dt * b_param[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            state = transition * state + drive
            y = (state * c_param[:, t].unsqueeze(1)).sum(-1)
            y = y + self.D.unsqueeze(0) * u[:, t]
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return residual + torch.sigmoid(self.gate) * y
