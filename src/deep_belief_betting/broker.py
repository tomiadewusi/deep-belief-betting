from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from deep_belief_betting.parameters import Parameters


class PositionSide(str, Enum):
    """Agent position side."""

    FLAT = "flat"
    YES = "yes"
    NO = "no"


@dataclass(frozen=True)
class BrokerState:
    """Current agent-side state."""

    has_entered: bool
    is_dead: bool
    side: PositionSide
    position_size: int
    entry_public_probability: float
    realised_cash_pnl: float
    last_trade_fee: float
    net_pnl_if_liquidated_now: float


class Broker:
    """Broker that handles LMSR execution and explicit fees."""

    def __init__(self, params: Parameters):
        self.params = params
        self._state: Optional[BrokerState] = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid."""
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    def _cost(self, q: float) -> float:
        """Return the LMSR cost function."""
        b = self.params.lmsr.b
        return b * float(np.log1p(np.exp(q / b)))

    def _execution_value(self, q: float, signed_yes_contracts: float) -> float:
        """Return gross LMSR execution value."""
        return self._cost(q + signed_yes_contracts) - self._cost(q)

    def _fee(self, gross_execution_value: float) -> float:
        """Return fixed plus proportional fee."""
        fixed_fee = self.params.fees.fixed_fee
        proportional_rate = self.params.fees.proportional_fee_bps / 10_000.0
        return fixed_fee + proportional_rate * abs(gross_execution_value)

    def reset(self) -> BrokerState:
        """Reset broker state at the start of an episode."""
        self._state = BrokerState(
            has_entered=False,
            is_dead=False,
            side=PositionSide.FLAT,
            position_size=0,
            entry_public_probability=0.0,
            realised_cash_pnl=0.0,
            last_trade_fee=0.0,
            net_pnl_if_liquidated_now=0.0,
        )
        return self._state

    def get_state(self) -> BrokerState:
        """Return the current broker state."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")
        return self._state

    def trade_size(self) -> int:
        """Return the fixed trade size for the episode."""
        return self.params.trade_size()

    def signed_entry_trade(self, side: PositionSide) -> float:
        """Return signed YES contracts for an entry trade."""
        size = self.trade_size()
        if side == PositionSide.YES:
            return float(size)
        if side == PositionSide.NO:
            return float(-size)
        raise ValueError("entry side must be yes or no")

    def signed_unwind_trade(self, side: PositionSide) -> float:
        """Return signed YES contracts for an unwind trade."""
        size = self.trade_size()
        if side == PositionSide.YES:
            return float(-size)
        if side == PositionSide.NO:
            return float(size)
        raise ValueError("cannot unwind a flat position")

    def preview_entry_cost(self, q: float, side: PositionSide) -> float:
        """Return net entry cost including explicit fees."""
        signed_trade = self.signed_entry_trade(side)
        gross_cost = self._execution_value(q, signed_trade)
        fee = self._fee(gross_cost)
        return gross_cost + fee

    def preview_unwind_value(self, q: float, side: PositionSide) -> float:
        """Return net unwind cash inflow after explicit fees."""
        signed_trade = self.signed_unwind_trade(side)
        gross_value = -self._execution_value(q, signed_trade)

        # gross value is positive cash back to the agent
        fee = self._fee(gross_value)
        return gross_value - fee

    def enter(self, q: float, public_probability: float, side: PositionSide) -> float:
        """Enter a fixed YES or NO position and return signed YES trade."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")
        if self._state.has_entered:
            raise RuntimeError("single roundtrip broker cannot re enter")

        signed_trade = self.signed_entry_trade(side)
        gross_cost = self._execution_value(q, signed_trade)
        fee = self._fee(gross_cost)
        cash_change = -(gross_cost + fee)

        # realized cash pnl records actual money moved
        new_realised_cash_pnl = self._state.realised_cash_pnl + cash_change

        self._state = BrokerState(
            has_entered=True,
            is_dead=False,
            side=side,
            position_size=self.trade_size(),
            entry_public_probability=public_probability,
            realised_cash_pnl=new_realised_cash_pnl,
            last_trade_fee=fee,
            net_pnl_if_liquidated_now=0.0,
        )
        return signed_trade

    def exit(self, q: float) -> float:
        """Exit the current position and return signed YES trade."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")
        if not self._state.has_entered or self._state.is_dead:
            raise RuntimeError("cannot exit without a live position")
        if self._state.side == PositionSide.FLAT:
            raise RuntimeError("cannot exit a flat position")

        signed_trade = self.signed_unwind_trade(self._state.side)
        gross_value = -self._execution_value(q, signed_trade)
        fee = self._fee(gross_value)
        cash_change = gross_value - fee

        new_realised_cash_pnl = self._state.realised_cash_pnl + cash_change

        self._state = BrokerState(
            has_entered=not self.params.trade.allow_reentry,
            is_dead=not self.params.trade.allow_reentry,
            side=PositionSide.FLAT,
            position_size=0,
            entry_public_probability=self._state.entry_public_probability,
            realised_cash_pnl=new_realised_cash_pnl,
            last_trade_fee=fee,
            net_pnl_if_liquidated_now=new_realised_cash_pnl,
        )
        return signed_trade

    def update_mark_to_liquidation(self, q: float) -> None:
        """Refresh net pnl if liquidated now."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")

        if not self._state.has_entered or self._state.is_dead or self._state.side == PositionSide.FLAT:
            net_pnl = self._state.realised_cash_pnl
        else:
            unwind_value = self.preview_unwind_value(q, self._state.side)
            net_pnl = self._state.realised_cash_pnl + unwind_value

        self._state = BrokerState(
            has_entered=self._state.has_entered,
            is_dead=self._state.is_dead,
            side=self._state.side,
            position_size=self._state.position_size,
            entry_public_probability=self._state.entry_public_probability,
            realised_cash_pnl=self._state.realised_cash_pnl,
            last_trade_fee=self._state.last_trade_fee,
            net_pnl_if_liquidated_now=net_pnl,
        )

    def terminal_settlement(self, terminal_outcome: int) -> float:
        """Apply terminal settlement and return terminal cashflow."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")

        settlement = 0.0
        if self._state.has_entered and not self._state.is_dead:
            if self._state.side == PositionSide.YES:
                settlement = float(self._state.position_size * terminal_outcome)
            elif self._state.side == PositionSide.NO:
                settlement = float(self._state.position_size * (1 - terminal_outcome))

        new_realised_cash_pnl = self._state.realised_cash_pnl + settlement

        self._state = BrokerState(
            has_entered=self._state.has_entered,
            is_dead=True,
            side=PositionSide.FLAT,
            position_size=0,
            entry_public_probability=self._state.entry_public_probability,
            realised_cash_pnl=new_realised_cash_pnl,
            last_trade_fee=self._state.last_trade_fee,
            net_pnl_if_liquidated_now=new_realised_cash_pnl,
        )
        return settlement

    def action_mask(self) -> np.ndarray:
        """Return the valid action mask."""
        if self._state is None:
            raise RuntimeError("broker must be reset before use")

        # action order is | hold | buy_yes | buy_no | sell |
        if self._state.is_dead:
            return np.array([1, 0, 0, 0], dtype=np.int8)

        if not self._state.has_entered:
            return np.array([1, 1, 1, 0], dtype=np.int8)

        return np.array([1, 0, 0, 1], dtype=np.int8)