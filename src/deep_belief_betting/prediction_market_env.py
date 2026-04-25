from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from deep_belief_betting.broker import Broker, PositionSide
from deep_belief_betting.market_sim import MarketSim, MarketState
from deep_belief_betting.parameters import Parameters


class PredictionMarketEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for the single roundtrip prediction market problem."""

    metadata = {"render_modes": []}

    ACTION_HOLD = 0
    ACTION_BUY_YES = 1
    ACTION_BUY_NO = 2
    ACTION_SELL = 3

    def __init__(self, params: Parameters, belief_dim: int = 0):
        self.params = params
        self.belief_dim = belief_dim

        self.market = MarketSim(params)
        self.broker = Broker(params)

        # keep this unbounded for simplicity Mahdu can opine if we should add contraints 
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_dim(),),
            dtype=np.float32,
        )

        self._belief_vector = np.zeros(self.belief_dim, dtype=np.float32)
        self._episode_done = False

    def _observation_dim(self) -> int:
        """Compute observation dimension from feature toggles."""
        dim = 0
        f = self.params.features

        dim += int(f.include_public_probability)
        dim += int(f.include_same_day_flow)
        dim += int(f.include_time_to_resolution)
        dim += int(f.include_position_flag)
        dim += int(f.include_position_side)
        dim += int(f.include_cash_pnl)
        dim += int(f.include_entry_probability)
        dim += int(f.include_entry_costs_when_flat) * 2
        dim += int(f.include_unwind_value_when_invested)
        dim += int(f.explicit_dead_state)
        dim += self.belief_dim if self.params.belief_features.enabled and self.params.belief_features.mode == "vector" else 0
        return dim

    def set_belief_vector(self, belief_vector: np.ndarray) -> None:
        """Set the external pretrained belief vector for the current step."""
        if belief_vector.shape != (self.belief_dim,):
            raise ValueError("belief vector has wrong shape")
        self._belief_vector = belief_vector.astype(np.float32, copy=True)

    def _position_side_encoding(self) -> float:
        """Encode position side as a scalar."""
        side = self.broker.get_state().side
        if side == PositionSide.YES:
            return 1.0
        if side == PositionSide.NO:
            return -1.0
        return 0.0

    def _build_observation(self) -> np.ndarray:
        """Assemble the current observation vector."""
        market = self.market.get_state()
        broker = self.broker.get_state()
        f = self.params.features

        values: list[float] = []

        # market state
        if f.include_public_probability:
            values.append(float(market.public_probability))

        if f.include_same_day_flow:
            values.append(float(market.delta_q))

        if f.include_time_to_resolution:
            values.append(float(market.time_to_resolution))

        # agent state
        if f.include_position_flag:
            values.append(float(int(broker.has_entered and not broker.is_dead)))

        if f.include_position_side:
            values.append(float(self._position_side_encoding()))

        if f.include_cash_pnl:
            values.append(float(broker.realised_cash_pnl))

        if f.include_entry_probability:
            values.append(float(broker.entry_public_probability))

        # local execution economics
        if f.include_entry_costs_when_flat:
            if not broker.has_entered:
                buy_yes_cost = self.broker.preview_entry_cost(market.q, PositionSide.YES)
                buy_no_cost = self.broker.preview_entry_cost(market.q, PositionSide.NO)
                values.extend([float(buy_yes_cost), float(buy_no_cost)])
            else:
                values.extend([0.0, 0.0])

        if f.include_unwind_value_when_invested:
            if broker.has_entered and not broker.is_dead and broker.side != PositionSide.FLAT:
                unwind_value = self.broker.preview_unwind_value(market.q, broker.side)
                values.append(float(unwind_value))
            else:
                values.append(0.0)

        # explicit dead state
        if f.explicit_dead_state:
            values.append(float(int(broker.is_dead)))

        # pretrained belief features
        if self.params.belief_features.enabled and self.params.belief_features.mode == "vector":
            values.extend(self._belief_vector.astype(np.float32).tolist())

        return np.asarray(values, dtype=np.float32)

    def _reward_for_transition(
        self,
        realised_cash_change: float,
        terminal_settlement: float,
        invalid_action: bool,
    ) -> float:
        """Compute reward under the configured reward mode."""
        reward_mode = self.params.reward.reward_mode
        reward = 0.0

        if reward_mode == "realized_cashflow":
            reward += realised_cash_change + terminal_settlement

        elif reward_mode == "terminal_net_pnl":
            if self._episode_done:
                reward += self.broker.get_state().realised_cash_pnl

        # keep shaping tiny and explicit
        if invalid_action:
            reward -= self.params.reward.invalid_action_penalty

        return float(reward)

    def _handle_action(self, action: int) -> tuple[float, bool]:
        """Apply the agent action before exogenous flow."""
        broker_state = self.broker.get_state()
        market_state = self.market.get_state()

        action_mask = self.broker.action_mask()
        invalid_action = bool(action_mask[action] == 0)
        realised_cash_before = broker_state.realised_cash_pnl

        if invalid_action:
            return 0.0, True

        if action == self.ACTION_HOLD:
            return 0.0, False

        if action == self.ACTION_BUY_YES:
            signed_trade = self.broker.enter(
                q=market_state.q,
                public_probability=market_state.public_probability,
                side=PositionSide.YES,
            )
            self.market.apply_agent_trade(signed_trade)
            realised_cash_after = self.broker.get_state().realised_cash_pnl
            return realised_cash_after - realised_cash_before, False

        if action == self.ACTION_BUY_NO:
            signed_trade = self.broker.enter(
                q=market_state.q,
                public_probability=market_state.public_probability,
                side=PositionSide.NO,
            )
            self.market.apply_agent_trade(signed_trade)
            realised_cash_after = self.broker.get_state().realised_cash_pnl
            return realised_cash_after - realised_cash_before, False

        if action == self.ACTION_SELL:
            signed_trade = self.broker.exit(q=market_state.q)
            self.market.apply_agent_trade(signed_trade)
            realised_cash_after = self.broker.get_state().realised_cash_pnl
            return realised_cash_after - realised_cash_before, False

        raise ValueError("unknown action")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Reset one episode."""
        super().reset(seed=seed)
        self._episode_done = False

        self.market.reset(seed=seed)
        self.broker.reset()

        if self.belief_dim > 0:
            self._belief_vector = np.zeros(self.belief_dim, dtype=np.float32)

        obs = self._build_observation()
        info = {"action_mask": self.broker.action_mask().copy()}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one environment step."""
        if self._episode_done:
            raise RuntimeError("cannot step a finished environment")

        realised_cash_change, invalid_action = self._handle_action(action)

        # market evolves even in the dead state
        market_state, market_done = self.market.step()

        # refresh liquidation based diagnostics
        self.broker.update_mark_to_liquidation(market_state.q)

        terminal_settlement = 0.0
        if market_done:
            terminal_settlement = self.broker.terminal_settlement(market_state.terminal_outcome)
            self._episode_done = True

        reward = self._reward_for_transition(
            realised_cash_change=realised_cash_change,
            terminal_settlement=terminal_settlement,
            invalid_action=invalid_action,
        )

        obs = self._build_observation()
        info = {
            "action_mask": self.broker.action_mask().copy(),
            "terminal_outcome": market_state.terminal_outcome if market_done else None,
            "net_pnl_if_liquidated_now": self.broker.get_state().net_pnl_if_liquidated_now,
            "realised_cash_pnl": self.broker.get_state().realised_cash_pnl,
            "invalid_action": invalid_action,
        }
        terminated = self._episode_done
        truncated = False
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return the current action mask."""
        return self.broker.action_mask().copy()