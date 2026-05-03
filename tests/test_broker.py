from deep_belief_betting.broker import Broker, PositionSide
from deep_belief_betting.parameters import Parameters


def test_no_entry_is_true_long_no_cost() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    broker = Broker(params)
    broker.reset()

    q0 = 0.0
    yes_cost = broker.preview_entry_cost(q0, PositionSide.YES)
    no_cost = broker.preview_entry_cost(q0, PositionSide.NO)

    assert yes_cost > 0.0
    assert no_cost > 0.0
    assert abs(yes_cost - no_cost) < 1e-12


def test_buy_no_does_not_profit_when_yes_resolves() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    broker = Broker(params)
    broker.reset()

    broker.enter(q=0.0, public_probability=0.5, side=PositionSide.NO)
    broker.terminal_settlement(terminal_outcome=1)

    assert broker.get_state().realised_cash_pnl < 0.0


def test_immediate_no_roundtrip_loses_fees_only() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    broker = Broker(params)
    broker.reset()

    q0 = 0.0
    signed_entry = broker.enter(q=q0, public_probability=0.5, side=PositionSide.NO)
    signed_exit = broker.exit(q=q0 + signed_entry)

    assert signed_entry < 0.0
    assert signed_exit > 0.0
    assert broker.get_state().realised_cash_pnl < 0.0


def test_broker_entry_exit_and_reentry() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    broker = Broker(params)
    broker.reset()

    q0 = 0.0
    p0 = 0.5

    signed_trade = broker.enter(q=q0, public_probability=p0, side=PositionSide.YES)
    assert signed_trade > 0
    assert broker.get_state().has_entered is True
    assert broker.get_state().side == PositionSide.YES

    signed_exit = broker.exit(q=q0 + signed_trade)
    assert signed_exit < 0
    assert broker.get_state().is_dead is False
    assert broker.get_state().side == PositionSide.FLAT

    signed_reentry = broker.enter(q=q0, public_probability=p0, side=PositionSide.NO)
    assert signed_reentry < 0
    assert broker.get_state().side == PositionSide.NO
