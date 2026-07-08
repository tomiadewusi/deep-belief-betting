from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.pretraining_path_generator import PretrainingPathGenerator


def test_pretraining_episode_shapes() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    generator = PretrainingPathGenerator(params)

    episode = generator.generate_episode(seed=123)

    assert episode["features"].ndim == 2
    assert episode["features"].shape[1] == 3
    assert episode["terminal_label"].shape == (1,)
    assert episode["next_public_probability"].ndim == 1
    assert episode["next_flow_sign"].ndim == 1


def test_pretraining_dataset_shapes() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    generator = PretrainingPathGenerator(params)

    dataset = generator.generate_dataset(num_paths=4, base_seed=100)

    assert dataset["features"].shape[0] == 4
    assert dataset["terminal_label"].shape[0] == 4