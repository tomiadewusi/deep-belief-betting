import numpy as np

from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.prediction_market_env import PredictionMarketEnv


def test_env_reset_and_step() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    env = PredictionMarketEnv(params=params, belief_dim=0)

    obs, info = env.reset(seed=123)
    assert obs.ndim == 1
    assert "action_mask" in info
    assert info["action_mask"].shape == (4,)

    next_obs, reward, terminated, truncated, next_info = env.step(0)
    assert next_obs.ndim == 1
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in next_info

    env.close()


def test_env_action_mask_changes_after_entry() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    env = PredictionMarketEnv(params=params, belief_dim=0)

    _, info = env.reset(seed=123)
    assert np.array_equal(info["action_mask"], np.array([1, 0, 0, 0], dtype=np.int8))

    _, _, _, _, info = env.step(0)
    assert np.array_equal(info["action_mask"], np.array([1, 1, 1, 0], dtype=np.int8))

    _, _, _, _, info = env.step(1)
    assert np.array_equal(info["action_mask"], np.array([1, 0, 0, 1], dtype=np.int8))

    env.close()
