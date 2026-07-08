from deep_belief_betting.simulation.parameters import Parameters


def test_load_default_yaml() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    assert params.seed == 20260423
    assert params.num_steps == 60
    assert params.lmsr.b == 100.0
    assert params.trade_size() > 0