from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.pretraining_path_generator import PretrainingPathGenerator
from deep_belief_betting.belief_model.data import PriceDataset


def test_pretraining_episode_shapes() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    generator = PretrainingPathGenerator(params)

    episode = generator.generate_episode(seed=123)

    assert episode["features"].ndim == 2
    assert episode["features"].shape == (params.num_steps + 1, 2)
    assert episode["latent_probability"].shape == (params.num_steps + 1,)
    assert episode["terminal_label"].shape == (1,)
    assert episode["next_public_probability"].shape == (params.num_steps,)
    assert episode["next_flow_sign"].shape == (params.num_steps,)


def test_pretraining_dataset_shapes() -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    generator = PretrainingPathGenerator(params)

    dataset = generator.generate_dataset(num_paths=4, base_seed=100)

    assert dataset["features"].shape[0] == 4
    assert dataset["features"].shape[1:] == (params.num_steps + 1, 2)
    assert dataset["latent_probability"].shape == (4, params.num_steps + 1)
    assert dataset["terminal_label"].shape[0] == 4


def test_pretraining_csv_matches_price_dataset(tmp_path) -> None:
    params = Parameters.from_yaml("configs/default.yaml")
    generator = PretrainingPathGenerator(params)

    csv_path = generator.write_price_dataset_csv(
        tmp_path / "belief_paths.csv",
        num_paths=3,
        base_seed=200,
    )
    dataset = PriceDataset(csv_path=str(csv_path), T=params.num_steps)

    assert len(dataset) == 3
    features, targets, terminal_label = dataset[0]
    assert features.shape == (params.num_steps + 1, 2)
    assert targets.shape == (params.num_steps + 1,)
    assert terminal_label.ndim == 0
