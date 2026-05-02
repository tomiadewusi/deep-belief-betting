from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from model.device import resolve_device

REQUIRED_KEYS = [
    "T", "in_dim",
    "d_model", "n_heads", "n_layers", "d_ff", "dropout",
    "d_z",
    "d_dec_hidden",
    "batch_size",
    "lr", "weight_decay", "grad_clip",
    "n_epochs",
    "device", "seed",
    "data_path",
    "checkpoint_path",
]

OPTIONAL_KEYS_WITH_DEFAULTS = {
    "enable_true_prob_head": False,
    "true_prob_loss_weight": 0.5,
}


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)

    expected = set(REQUIRED_KEYS)
    optional = set(OPTIONAL_KEYS_WITH_DEFAULTS.keys())
    missing = expected - raw.keys()
    extra = raw.keys() - expected - optional
    if missing:
        raise KeyError(f"config missing keys: {sorted(missing)}")
    if extra:
        raise KeyError(f"config has unknown keys: {sorted(extra)}")

    for key, default in OPTIONAL_KEYS_WITH_DEFAULTS.items():
        raw.setdefault(key, default)

    cfg = SimpleNamespace(**raw)

    cfg.device = resolve_device("auto")
    return cfg


class PriceDataset(Dataset):
    """One row per training example.

    Expected CSV columns:
        prob_0, prob_1, ..., prob_T       (public probability)
        feat_0, feat_1, ..., feat_T       (flow variable)
        latent_0, latent_1, ..., latent_T (target: latent probability)
        terminal_label (final outcome of the episode)
    Each item is a tuple (features, targets):
        features: (T+1, 2) — stacked [prob_i, feat_i] per timestep
        targets:  (T+1,)   — latent probabilities, soft labels in [0,1]
        terminal_label: float 0 or 1
    """

    def __init__(self, csv_path: str, T: int):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path.resolve()}")

        df = pd.read_csv(path)

        prob_cols = [f"prob_{i}" for i in range(T + 1)]
        feat_cols = [f"feat_{i}" for i in range(T + 1)]
        latent_cols = [f"latent_{i}" for i in range(T + 1)]

        missing = [c for c in prob_cols + feat_cols + latent_cols + ["terminal_label"] if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing columns: {missing}")

        probs = torch.tensor(df[prob_cols].values, dtype=torch.float32)    # (N, T+1)
        feats = torch.tensor(df[feat_cols].values, dtype=torch.float32)    # (N, T+1)
        self.features = torch.stack([probs, feats], dim=-1)                # (N, T+1, 2)
        self.targets = torch.tensor(df[latent_cols].values, dtype=torch.float32)  # (N, T+1)
        self.terminal_labels = torch.tensor(df["terminal_label"].values, dtype=torch.float32) # (N,)
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        return self.features[i], self.targets[i], self.terminal_labels[i]