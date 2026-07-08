from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from model.model import Architecture3
from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.market_sim import MarketSim


def _remap_state_dict(sd: dict) -> dict:
    remapped = {}
    for k, v in sd.items():
        if k.startswith("decoder_mlp."):
            k = "latent_" + k
        remapped[k] = v
    return remapped


def load_model(checkpoint_path: str | Path, device: torch.device) -> Tuple[Architecture3, SimpleNamespace]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = SimpleNamespace(**ckpt["cfg"])
    model = Architecture3(cfg).to(device)
    model.load_state_dict(_remap_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model, cfg


def generate_eval_episodes(
    params: Parameters,
    n_episodes: int,
    T: int,
    base_seed: int = 99999,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = T + 1
    features = np.zeros((n_episodes, L, 2), dtype=np.float32)
    latent_targets = np.zeros((n_episodes, L), dtype=np.float32)
    terminal_labels = np.zeros(n_episodes, dtype=np.float32)

    for i in range(n_episodes):
        sim = MarketSim(params)
        state = sim.reset(seed=base_seed + i)

        step = 0
        done = False
        while not done:
            if step < L:
                features[i, step, 0] = state.public_probability
                features[i, step, 1] = state.delta_q
                latent_targets[i, step] = state.latent_probability
            state, done = sim.step()
            step += 1

        # Collect the terminal state (returned after done=True) if it fits in the window
        if step < L:
            features[i, step, 0] = state.public_probability
            features[i, step, 1] = state.delta_q
            latent_targets[i, step] = state.latent_probability
            step += 1

        actual = min(step, L)
        if actual < L:
            features[i, actual:] = features[i, actual - 1]
            latent_targets[i, actual:] = latent_targets[i, actual - 1]

        terminal_labels[i] = float(state.terminal_outcome)

    return features, latent_targets, terminal_labels


def run_model_on_episodes(
    model: Architecture3,
    features: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    N, L, _ = features.shape
    p_latent = np.zeros((N, L), dtype=np.float32)
    p_terminal = np.zeros((N, L), dtype=np.float32) if model.cfg.enable_true_prob_head else None
    z_final = np.zeros((N, model.cfg.d_z), dtype=np.float32)

    features_t = torch.tensor(features, dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in range(L):
            x = features_t[:, : t + 1, :]
            p_t, _, z_t, out_t = model(x)
            p_latent[:, t] = p_t.cpu().numpy()
            if p_terminal is not None and out_t is not None:
                p_terminal[:, t] = torch.sigmoid(out_t).cpu().numpy()
            if t == L - 1:
                z_final = z_t.cpu().numpy()

    return p_latent, p_terminal, z_final


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_sorted = y_true[order].astype(float)
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1.0 - y_sorted) / n_neg
    return float(np.trapz(tpr, fpr))


def compute_metrics(
    p_latent: np.ndarray,
    latent_targets: np.ndarray,
    p_terminal: Optional[np.ndarray],
    terminal_labels: np.ndarray,
) -> Dict:
    eps = 1e-7
    p = np.clip(p_latent, eps, 1 - eps)
    y = latent_targets

    bce_latent = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    mae_latent = float(np.abs(p - y).mean())
    mse_latent = float(((p - y) ** 2).mean())

    metrics: Dict = {
        "latent_head": {
            "bce": bce_latent,
            "mae": mae_latent,
            "mse": mse_latent,
        }
    }

    if p_terminal is not None:
        pt = np.clip(p_terminal, eps, 1 - eps)
        tl = np.broadcast_to(terminal_labels[:, None], p_terminal.shape)
        bce_term = float(-(tl * np.log(pt) + (1 - tl) * np.log(1 - pt)).mean())
        metrics["terminal_head"] = {
            "bce": bce_term,
            "perplexity": float(np.exp(bce_term)),
            "mae": float(np.abs(pt - tl).mean()),
            "mse": float(((pt - tl) ** 2).mean()),
            "auc_roc_final_step": _auc_roc(terminal_labels, p_terminal[:, -1]),
        }

    N, L = p_latent.shape
    sharpe_by_t = []
    for t in range(L):
        signal = 2.0 * (p_latent[:, t] - 0.5)
        returns = signal * (2.0 * terminal_labels - 1.0)
        sharpe_by_t.append(float(returns.mean() / (returns.std() + 1e-8)))

    metrics["sharpe_by_timestep"] = sharpe_by_t
    metrics["sharpe_final"] = sharpe_by_t[-1]
    metrics["sharpe_max"] = float(max(sharpe_by_t))
    metrics["sharpe_max_timestep"] = int(np.argmax(sharpe_by_t))

    bins = np.linspace(0, 1, 11)
    bin_centers = (0.5 * (bins[:-1] + bins[1:])).tolist()
    bin_idx = np.clip(np.digitize(p_latent.ravel(), bins) - 1, 0, 9)
    empirical = []
    for b in range(10):
        mask = bin_idx == b
        empirical.append(float(latent_targets.ravel()[mask].mean()) if mask.sum() > 0 else float("nan"))
    metrics["calibration"] = {"bin_centers": bin_centers, "empirical_latent": empirical}

    return metrics


def evaluate(
    checkpoint_path: str | Path,
    world_config_path: str | Path = "configs/default.yaml",
    n_episodes: int = 500,
    output_path: Optional[str | Path] = None,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    model, cfg = load_model(checkpoint_path, device)
    params = Parameters.from_yaml(world_config_path)
    features, latent_targets, terminal_labels = generate_eval_episodes(params, n_episodes, cfg.T)
    p_latent, p_terminal, z_final = run_model_on_episodes(model, features, device)
    metrics = compute_metrics(p_latent, latent_targets, p_terminal, terminal_labels)

    lh = metrics["latent_head"]
    print(f"\n=== Belief Model Evaluation ({n_episodes} episodes) ===")
    print(f"Latent head  | BCE: {lh['bce']:.4f}  MAE: {lh['mae']:.4f}  MSE: {lh['mse']:.4f}")
    if "terminal_head" in metrics:
        th = metrics["terminal_head"]
        print(
            f"Terminal head| BCE: {th['bce']:.4f}  Perplexity: {th['perplexity']:.4f}"
            f"  MAE: {th['mae']:.4f}  MSE: {th['mse']:.4f}  AUC: {th['auc_roc_final_step']:.4f}"
        )
    print(f"Sharpe (final step): {metrics['sharpe_final']:.4f}")
    print(f"Sharpe (max t={metrics['sharpe_max_timestep']}): {metrics['sharpe_max']:.4f}")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved → {out}")

    return metrics, features, latent_targets, terminal_labels, p_latent, p_terminal, z_final
