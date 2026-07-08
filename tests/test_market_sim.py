from deep_belief_betting.simulation.market_sim import MarketSim
from deep_belief_betting.simulation.parameters import Parameters


def test_market_sim_reset_and_step() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    sim = MarketSim(params)

    state = sim.reset(seed=123)
    assert state.step == 0
    assert 0.0 < state.public_probability < 1.0

    next_state, done = sim.step()
    assert next_state.step == 1
    assert isinstance(done, bool)
    assert 0.0 < next_state.public_probability < 1.0