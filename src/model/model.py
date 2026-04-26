
import math
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.p_drop = dropout

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x):
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self._split, (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.p_drop if self.training else 0.0,
            is_causal=True,
        )
        return self.out(self._merge(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Architecture3(nn.Module):
    """
    Inputs:
        x: (B, T+1, 2) where each slot is (price, prev_prob).
    Outputs:
        p_t:    (B,)         binary classification probability in [0, 1]
        logit:  (B,)         pre-sigmoid logit (use BCEWithLogitsLoss)
        z_t:    (B, d_z)     state vector
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        L_in = cfg.T + 1

        self.input_proj = nn.Linear(cfg.in_dim, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, max_len=max(L_in * 4, 512)
        )
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.d_model)

        self.state_proj = nn.Linear(cfg.d_model, cfg.d_z)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(cfg.d_z, cfg.d_dec_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_dec_hidden, cfg.d_dec_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_dec_hidden, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, in_dim = x.shape
        assert L <= self.cfg.T + 1, f"sequence length {L} exceeds T+1={self.cfg.T+1}"
        assert in_dim == self.cfg.in_dim, f"expected in_dim={self.cfg.in_dim}, got {in_dim}"

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.input_dropout(h)

        for layer in self.encoder_layers:
            h = layer(h)
        h = self.encoder_norm(h)

        z_t = self.state_proj(h[:, -1, :])

        logit = self.decoder_mlp(z_t).squeeze(-1)
        p_t = torch.sigmoid(logit)
        return p_t, logit, z_t
