from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from deep_belief_betting.belief_model.model import Architecture3


def _pca_2d(z: np.ndarray):
    z_c = z - z.mean(0)
    _, s, Vt = np.linalg.svd(z_c.T @ z_c / len(z_c))
    z2d = z_c @ Vt[:2].T
    var = s[:2] / s.sum()
    return z2d, var


def plot_attention_heatmaps(
    model: Architecture3,
    features: np.ndarray,
    episode_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
):
    model.eval()
    L = model.cfg.T + 1
    x = torch.tensor(features[episode_idx : episode_idx + 1, :L, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, _, _, all_attn = model.forward_with_attn(x)

    n_layers = len(all_attn)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for li, attn in enumerate(all_attn):
        avg = attn[0].mean(0).cpu().numpy()
        im = axes[li].imshow(avg, aspect="auto", cmap="viridis")
        axes[li].set_title(f"Layer {li + 1}")
        axes[li].set_xlabel("Key t")
        axes[li].set_ylabel("Query t")
        plt.colorbar(im, ax=axes[li], fraction=0.046)

    fig.suptitle(f"Attention Heatmaps (avg heads) — Episode {episode_idx}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_head_attention(
    model: Architecture3,
    features: np.ndarray,
    episode_idx: int = 0,
    layer_idx: int = 0,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
):
    model.eval()
    L = model.cfg.T + 1
    n_heads = model.cfg.n_heads
    x = torch.tensor(features[episode_idx : episode_idx + 1, :L, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, _, _, all_attn = model.forward_with_attn(x)

    attn = all_attn[layer_idx][0].cpu().numpy()
    cols = min(n_heads, 4)
    rows = math.ceil(n_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).ravel()

    for h in range(n_heads):
        im = axes[h].imshow(attn[h], aspect="auto", cmap="viridis")
        axes[h].set_title(f"Head {h + 1}")
        axes[h].set_xlabel("Key t")
        axes[h].set_ylabel("Query t")
        plt.colorbar(im, ax=axes[h], fraction=0.046)
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(f"Per-Head Attention — Layer {layer_idx + 1}, Episode {episode_idx}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_entropy(
    model: Architecture3,
    features: np.ndarray,
    n_episodes: int = 50,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
):
    model.eval()
    L = model.cfg.T + 1
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    entropy_sum = np.zeros((n_layers, n_heads))
    count = 0

    for ep in range(min(n_episodes, len(features))):
        x = torch.tensor(features[ep : ep + 1, :L, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            _, _, _, _, all_attn = model.forward_with_attn(x)
        for li, attn in enumerate(all_attn):
            a = np.clip(attn[0].cpu().numpy(), 1e-9, 1.0)
            entropy_sum[li] += (-(a * np.log(a)).sum(-1)).mean(-1)
        count += 1

    entropy_mean = entropy_sum / count
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 1.2), 4))
    x_pos = np.arange(n_heads)
    width = 0.8 / n_layers
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_layers))
    for li in range(n_layers):
        ax.bar(x_pos + li * width, entropy_mean[li], width, label=f"Layer {li + 1}", color=colors[li], alpha=0.85)
    ax.set_xlabel("Head")
    ax.set_ylabel("Mean Entropy (nats)")
    ax.set_title("Attention Entropy per Head per Layer")
    ax.set_xticks(x_pos + width * (n_layers - 1) / 2)
    ax.set_xticklabels([f"H{h + 1}" for h in range(n_heads)])
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sharpe_vs_time(
    sharpe_by_t: List[float],
    save_path: Optional[Path] = None,
):
    T = len(sharpe_by_t)
    s = np.array(sharpe_by_t)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(T), s, linewidth=2, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(range(T), 0, s, where=s > 0, alpha=0.2, color="green", label="Positive")
    ax.fill_between(range(T), 0, s, where=s <= 0, alpha=0.2, color="red", label="Negative")
    peak = int(np.argmax(s))
    ax.axvline(peak, color="orange", linewidth=1.5, linestyle=":", label=f"Peak t={peak} ({s[peak]:.3f})")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio of Latent Signal vs Timestep")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(
    metrics: dict,
    save_path: Optional[Path] = None,
):
    cal = metrics["calibration"]
    centers = np.array(cal["bin_centers"])
    empirical = np.array(cal["empirical_latent"])
    valid = ~np.isnan(empirical)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(centers[valid], empirical[valid], width=0.08, alpha=0.4, color="steelblue")
    ax.plot(centers[valid], empirical[valid], "o-", color="steelblue", label="Empirical")
    ax.set_xlabel("Predicted p_t")
    ax.set_ylabel("Mean empirical latent probability")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_belief_pca(
    z_final: np.ndarray,
    terminal_labels: np.ndarray,
    save_path: Optional[Path] = None,
):
    z2d, var = _pca_2d(z_final)
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in [(0, "crimson", "NO (0)"), (1, "steelblue", "YES (1)")]:
        mask = terminal_labels == label
        ax.scatter(z2d[mask, 0], z2d[mask, 1], c=color, label=name, alpha=0.5, s=20)
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
    ax.set_title("PCA of Final Belief Vectors z_T")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
