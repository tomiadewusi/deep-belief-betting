from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from deep_belief_betting.simulation.market_sim import MarketSim
from deep_belief_betting.simulation.parameters import Parameters


class PretrainingPathGenerator:
    """Generate supervised path data for encoder decoder pretraining."""

    def __init__(self, params: Parameters):
        self.params = params

    def _feature_vector(self, market_state) -> np.ndarray:
        """Build one market only feature vector for pretraining."""
        return np.asarray(
            [
                market_state.public_probability,
                market_state.delta_q,
            ],
            dtype=np.float32,
        )

    def generate_episode(
        self,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate one path and return transition aligned features and labels."""
        sim = MarketSim(self.params)
        state = sim.reset(seed=seed)

        features: List[np.ndarray] = [self._feature_vector(state)]
        latent_probability_targets: List[float] = [float(state.latent_probability)]
        public_prob_targets: List[float] = []
        flow_sign_targets: List[float] = []

        done = False
        while not done:
            next_state, done = sim.step()

            public_prob_targets.append(float(next_state.public_probability))
            flow_sign_targets.append(float(np.sign(next_state.delta_q)))
            features.append(self._feature_vector(next_state))
            latent_probability_targets.append(float(next_state.latent_probability))

            state = next_state

        terminal_label = float(state.terminal_outcome)

        return {
            "features": np.stack(features, axis=0),
            "latent_probability": np.asarray(latent_probability_targets, dtype=np.float32),
            "terminal_label": np.asarray([terminal_label], dtype=np.float32),
            "next_public_probability": np.asarray(public_prob_targets, dtype=np.float32),
            "next_flow_sign": np.asarray(flow_sign_targets, dtype=np.float32),
        }

    def generate_dataset(
        self,
        num_paths: int,
        base_seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate a padded dataset of fixed length paths."""
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")

        base_seed = self.params.seed if base_seed is None else base_seed

        feature_batch: List[np.ndarray] = []
        label_batch: List[np.ndarray] = []
        latent_probability_batch: List[np.ndarray] = []
        next_public_prob_batch: List[np.ndarray] = []
        next_flow_sign_batch: List[np.ndarray] = []

        for i in range(num_paths):
            episode = self.generate_episode(seed=base_seed + i)

            feature_batch.append(episode["features"])
            label_batch.append(episode["terminal_label"])
            latent_probability_batch.append(episode["latent_probability"])
            next_public_prob_batch.append(episode["next_public_probability"])
            next_flow_sign_batch.append(episode["next_flow_sign"])

        return {
            "features": np.stack(feature_batch, axis=0),
            "latent_probability": np.stack(latent_probability_batch, axis=0),
            "terminal_label": np.stack(label_batch, axis=0),
            "next_public_probability": np.stack(next_public_prob_batch, axis=0),
            "next_flow_sign": np.stack(next_flow_sign_batch, axis=0),
        }

    def write_price_dataset_csv(
        self,
        path: str | Path,
        num_paths: int,
        base_seed: Optional[int] = None,
    ) -> Path:
        """Write a CSV compatible with ``belief_model.data.PriceDataset``."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = self.generate_dataset(num_paths=num_paths, base_seed=base_seed)
        features = dataset["features"]
        latent_probability = dataset["latent_probability"]
        terminal_label = dataset["terminal_label"].reshape(num_paths)
        horizon = features.shape[1] - 1

        fieldnames = (
            [f"prob_{i}" for i in range(horizon + 1)]
            + [f"feat_{i}" for i in range(horizon + 1)]
            + [f"latent_{i}" for i in range(horizon + 1)]
            + ["terminal_label"]
        )

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row_idx in range(num_paths):
                row: dict[str, float] = {}
                for t in range(horizon + 1):
                    row[f"prob_{t}"] = float(features[row_idx, t, 0])
                    row[f"feat_{t}"] = float(features[row_idx, t, 1])
                    row[f"latent_{t}"] = float(latent_probability[row_idx, t])
                row["terminal_label"] = float(terminal_label[row_idx])
                writer.writerow(row)

        return output_path
