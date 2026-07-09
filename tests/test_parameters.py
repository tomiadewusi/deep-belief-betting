from deep_belief_betting.simulation.parameters import Parameters


def test_load_default_yaml() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    assert params.seed == 20260424
    assert params.num_steps == 64
    assert params.lmsr.b == 20.0
    assert params.trade.allow_reentry is True
    assert params.trade_size() > 0
