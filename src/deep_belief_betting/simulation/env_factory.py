from __future__ import annotations

from typing import Callable

import gymnasium as gym

from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.prediction_market_env import PredictionMarketEnv


def make_env(params: Parameters, rank: int, belief_dim: int = 0) -> Callable[[], PredictionMarketEnv]:
    """Return a thunk that builds one environment instance."""

    def _thunk() -> PredictionMarketEnv:
        # keep seed offset deterministic across workers
        env = PredictionMarketEnv(params=params, belief_dim=belief_dim)
        env.reset(seed=params.seed + rank)
        return env

    return _thunk


def make_vector_env(
    params: Parameters,
    num_envs: int,
    belief_dim: int = 0,
) -> gym.vector.SyncVectorEnv:
    """Build a sync vectorized environment."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    env_fns = [make_env(params=params, rank=i, belief_dim=belief_dim) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)