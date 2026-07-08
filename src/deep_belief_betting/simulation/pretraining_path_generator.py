from __future__ import annotations

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
        # keep these raw and simple
        # the model can learn its own representation
        return np.asarray(
            [
                market_state.public_probability,
                market_state.delta_q,
                market_state.time_to_resolution,
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

        features: List[np.ndarray] = []
        public_prob_targets: List[float] = []
        flow_sign_targets: List[float] = []

        done = False
        while not done:
            current_state = state

            # step once to get the one step ahead targets
            next_state, done = sim.step()

            # feature is built from time t
            features.append(self._feature_vector(current_state))

            # auxiliary targets are built from time t plus 1
            public_prob_targets.append(float(next_state.public_probability))
            flow_sign_targets.append(float(np.sign(next_state.delta_q)))

            # advance the state pointer
            state = next_state

        terminal_label = float(state.terminal_outcome)

        return {
            "features": np.stack(features, axis=0),
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
        next_public_prob_batch: List[np.ndarray] = []
        next_flow_sign_batch: List[np.ndarray] = []

        for i in range(num_paths):
            episode = self.generate_episode(seed=base_seed + i)

            feature_batch.append(episode["features"])
            label_batch.append(episode["terminal_label"])
            next_public_prob_batch.append(episode["next_public_probability"])
            next_flow_sign_batch.append(episode["next_flow_sign"])

        return {
            "features": np.stack(feature_batch, axis=0),
            "terminal_label": np.stack(label_batch, axis=0),
            "next_public_probability": np.stack(next_public_prob_batch, axis=0),
            "next_flow_sign": np.stack(next_flow_sign_batch, axis=0),
        }