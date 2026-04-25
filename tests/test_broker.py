from deep_belief_betting.broker import Broker, PositionSide
from deep_belief_betting.parameters import Parameters


def test_broker_entry_and_exit() -> None:
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
    assert broker.get_state().is_dead is True
    assert broker.get_state().side == PositionSide.FLAT