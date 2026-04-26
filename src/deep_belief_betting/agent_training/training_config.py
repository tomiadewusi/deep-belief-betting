from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import yaml

@dataclass(frozen=True)
class TrainingConfig:
    #env
    seed: int
    num_envs: int # how many parallel environments to run
    num_updates: int # how many policy updates to perform
    rollout_steps: int #for each policy update, how many steps of experience should we collect?

    device: Literal["auto", "cpu", "cuda", "mps"]
    log_dir: str
    run_name: str
    save_interval: int
    log_interval: int

    #base world
    world_yaml_base_path: str

    #policy
    hidden_dim: int
    num_layers: int
    lr: float
    max_grad_norm: float

    #PPO train hyperparams
    gamma: float
    gae_lambda: float #trade off bias vs. variance in adv. estimate
    clip_range: float #stable updates, determines the maximum allowed policy update
    entropy_coef: float #encourages policy to explore by encouraging higher entropy
    value_coef: float #determines weight of value estimator loss in the loss function
    ppo_epochs: int #number of times same chosen batch of data is used to update the policy
    num_minibatches: int #how many mini-batches should we split the collected experience into?

    #belief feature
    belief_on: bool
    belief_dim: int #length of belief vector
    belief_checkpoint_path: str #path to load belief model from
    use_tensorboard: bool = True


def load_training_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return TrainingConfig(**config)



