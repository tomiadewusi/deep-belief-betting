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
        self._belief_q: float = 0.5
        self._belief_terminal: float = 0.5
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
        dim += int(f.include_belief_q)
        dim += int(f.include_belief_terminal)
        return dim

    def set_belief_vector(self, belief_vector: np.ndarray) -> None:
        """Set the external pretrained belief vector for the current step."""
        if belief_vector.shape != (self.belief_dim,):
            raise ValueError("belief vector has wrong shape")
        self._belief_vector = belief_vector.astype(np.float32, copy=True)

    def set_belief_q(self, q: float) -> None:
        """Set the scalar belief probability predicted by the pretrained model."""
        self._belief_q = float(q)

    def set_belief_terminal(self, terminal_probability: float) -> None:
        """Set the scalar terminal outcome probability from the pretrained model."""
        self._belief_terminal = float(terminal_probability)

    def _position_side_encoding(self) -> float:
        """Encode position side as a scalar."""
        side = self.broker.get_state().side
        if side == PositionSide.YES:
            return 1.0
        if side == PositionSide.NO:
            return -1.0
        return 0.0

    def _build_observation(self) -> np.ndarray:
        """Assemble the normalized agent-facing observation vector."""
        raw_observation = self._build_raw_observation()
        return self._normalize_raw_observation(raw_observation)

    def refresh_observation(self) -> np.ndarray:
        """Rebuild the current normalized observation with latest external beliefs."""
        return self._build_observation()


    def _center_probability(self, probability: float) -> float:
        """Map probability from [0, 1] to [-1, 1]."""
        return 2.0 * float(probability) - 1.0


    def _center_binary(self, value: bool | int | float) -> float:
        """Map binary feature from {0, 1} to {-1, 1}."""
        return 2.0 * float(value) - 1.0


    def _center_time_to_resolution(self, time_to_resolution: float) -> float:
        """Map remaining time fraction from [0, 1] to [-1, 1]."""
        fraction = float(time_to_resolution) / float(self.params.horizon_days)
        return 2.0 * fraction - 1.0


    def _scale_and_clip_by_liquidity(self, value: float) -> float:
        """Scale unbounded real-valued economic features by LMSR liquidity."""
        scaled = float(value) / float(self.params.lmsr.b)
        clip = float(self.params.observation_normalization.clip_value)
        return float(np.clip(scaled, -clip, clip))


    def _normalize_observation_value(self, name: str, value: float) -> float:
        """Normalize one raw observation feature."""
        if not self.params.observation_normalization.enabled:
            return float(value)

        if name in {"public_probability", "entry_public_probability"}:
            return self._center_probability(value)

        if name == "time_to_resolution":
            return self._center_time_to_resolution(value)

        if name in {"position_flag", "explicit_dead_state"}:
            return self._center_binary(value)

        if name == "position_side":
            return float(value)

        if name in {
            "same_day_flow",
            "cash_pnl",
            "entry_cost_yes",
            "entry_cost_no",
            "unwind_value",
        }:
            return self._scale_and_clip_by_liquidity(value)

        if name.startswith("belief_"):
            return float(value)

        raise ValueError(f"unknown observation feature: {name}")


    def _normalize_raw_observation(self, raw_observation: dict[str, float]) -> np.ndarray:
        """Convert ordered raw observation features into the agent-facing vector."""
        values = [
            self._normalize_observation_value(name, value)
            for name, value in raw_observation.items()
        ]
        return np.asarray(values, dtype=np.float32)


    def _build_raw_observation(self) -> dict[str, float]:
        """Assemble named raw observation features before normalization."""
        market = self.market.get_state()
        broker = self.broker.get_state()
        f = self.params.features

        values: dict[str, float] = {}

        if f.include_public_probability:
            values["public_probability"] = float(market.public_probability)

        if f.include_same_day_flow:
            values["same_day_flow"] = float(market.delta_q)

        if f.include_time_to_resolution:
            values["time_to_resolution"] = float(market.time_to_resolution)

        if f.include_position_flag:
            values["position_flag"] = float(int(broker.has_entered and not broker.is_dead))

        if f.include_position_side:
            values["position_side"] = float(self._position_side_encoding())

        if f.include_cash_pnl:
            values["cash_pnl"] = float(broker.realised_cash_pnl)

        if f.include_entry_probability:
            values["entry_public_probability"] = float(broker.entry_public_probability)

        if f.include_entry_costs_when_flat:
            if not broker.has_entered:
                values["entry_cost_yes"] = float(
                    self.broker.preview_entry_cost(market.q, PositionSide.YES)
                )
                values["entry_cost_no"] = float(
                    self.broker.preview_entry_cost(market.q, PositionSide.NO)
                )
            else:
                values["entry_cost_yes"] = 0.0
                values["entry_cost_no"] = 0.0

        if f.include_unwind_value_when_invested:
            if broker.has_entered and not broker.is_dead and broker.side != PositionSide.FLAT:
                values["unwind_value"] = float(
                    self.broker.preview_unwind_value(market.q, broker.side)
                )
            else:
                values["unwind_value"] = 0.0

        if f.explicit_dead_state:
            values["explicit_dead_state"] = float(int(broker.is_dead))

        if self.params.belief_features.enabled and self.params.belief_features.mode == "vector":
            for idx, belief_value in enumerate(self._belief_vector.astype(np.float32).tolist()):
                values[f"belief_{idx}"] = float(belief_value)

        if f.include_belief_q:
            values["belief_q"] = self._belief_q

        if f.include_belief_terminal:
            values["belief_terminal"] = self._belief_terminal

        return values



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

    def new_action_mask(self) -> np.ndarray:
        """Don't let the agent trade on timestep 0"""
        mask = self.broker.action_mask().astype(np.int8, copy=True)
        if self.market.get_state().step == 0:
            mask[1] = 0  # BUY_YES
            mask[2] = 0  # BUY_NO
            mask[3] = 0  # SELL
        return mask

    def _handle_action(self, action: int) -> tuple[float, bool]:
        """Apply the agent action before exogenous flow."""
        broker_state = self.broker.get_state()
        market_state = self.market.get_state()

        action_mask = self.new_action_mask()
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
        self._belief_q = 0.5
        self._belief_terminal = 0.5

        raw_observation = self._build_raw_observation()
        obs = self._normalize_raw_observation(raw_observation)
        info = {
            "action_mask": self.new_action_mask().copy(),
            "realised_cash_pnl": 0.0,
            "raw_observation": raw_observation,
            }
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
        raw_observation = self._build_raw_observation()
        obs = self._normalize_raw_observation(raw_observation)
        info = {
            "action_mask": self.new_action_mask().copy(),
            "terminal_outcome": market_state.terminal_outcome if market_done else None,
            "net_pnl_if_liquidated_now": self.broker.get_state().net_pnl_if_liquidated_now,
            "realised_cash_pnl": self.broker.get_state().realised_cash_pnl,
            "invalid_action": invalid_action,
            "raw_observation": raw_observation,
        }
        terminated = self._episode_done
        truncated = False
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return the current action mask."""
        return self.new_action_mask().copy()