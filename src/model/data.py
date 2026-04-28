from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


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


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)

    expected = set(REQUIRED_KEYS)
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        raise KeyError(f"config missing keys: {sorted(missing)}")
    if extra:
        raise KeyError(f"config has unknown keys: {sorted(extra)}")

    cfg = SimpleNamespace(**raw)

    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


class PriceDataset(Dataset):
    """One row per training example.

    Expected CSV columns:
        prob_0, prob_1, ..., prob_T       (public probability)
        feat_0, feat_1, ..., feat_T       (flow variable)
        latent_0, latent_1, ..., latent_T (target: latent probability)

    Each item is a tuple (features, targets):
        features: (T+1, 2) — stacked [prob_i, feat_i] per timestep
        targets:  (T+1,)   — latent probabilities, soft labels in [0,1]
    """

    def __init__(self, csv_path: str, T: int):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path.resolve()}")

        df = pd.read_csv(path)

        prob_cols = [f"prob_{i}" for i in range(T + 1)]
        feat_cols = [f"feat_{i}" for i in range(T + 1)]
        latent_cols = [f"latent_{i}" for i in range(T + 1)]

        missing = [c for c in prob_cols + feat_cols + latent_cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing columns: {missing}")

        probs = torch.tensor(df[prob_cols].values, dtype=torch.float32)    # (N, T+1)
        feats = torch.tensor(df[feat_cols].values, dtype=torch.float32)    # (N, T+1)
        self.features = torch.stack([probs, feats], dim=-1)                # (N, T+1, 2)
        self.targets = torch.tensor(df[latent_cols].values, dtype=torch.float32)  # (N, T+1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        return self.features[i], self.targets[i]