from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from deep_belief_betting.env_factory import make_vector_env
from deep_belief_betting.parameters import Parameters
from deep_belief_betting.agent_training.algorithms.ppo import compute_gae,compute_ppo_minibatch_loss

from deep_belief_betting.agent_training.device import resolve_device
from deep_belief_betting.agent_training.policy.mlp_actor_critic import ActorCritic
from deep_belief_betting.agent_training.run_layout import create_run_dir
from deep_belief_betting.agent_training.training_config import TrainingConfig, load_training_config


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_action_sample(
    logits: torch.Tensor, action_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions and log-probs (mask invalid logits)."""
    masked = logits.masked_fill(action_mask == 0, float('-inf'))
    dist = Categorical(logits=masked)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs


def train_ppo(cfg: TrainingConfig, device: torch.device) -> Path:
    set_seed(cfg.seed)
    run_dir = create_run_dir(cfg.log_dir, cfg.run_name, cfg)
    print(f"run_dir={run_dir}")

    params = Parameters.from_yaml(cfg.world_yaml_base_path)
    belief_dim = int(cfg.belief_dim) if cfg.belief_on else 0
    env = make_vector_env(params, cfg.num_envs, belief_dim=belief_dim)

    try:
        obs_dim = int(env.single_observation_space.shape[0])
        num_actions = int(env.single_action_space.n)

        model = ActorCritic(
            obs_dim,
            num_actions,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            device=device,
        )
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        n_envs = cfg.num_envs

        obs, infos = env.reset(seed=cfg.seed)

        #number of policy updates
        for update_idx in range(cfg.num_updates):
            rewards, dones, values, old_log_probs, actions, masks, observations = [], [], [], [], [], [], []
            
            #actual policy rollout for experience collection - happens in parallel across envs
            for i in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                mask_t = torch.as_tensor(np.asarray(infos["action_mask"], dtype=np.int64), device=device)

                with torch.no_grad():
                    logits, vals = model(obs_t)
                    action_t, logprobs_t = _masked_action_sample(logits, mask_t)

                observations.append(np.asarray(obs, dtype=np.float32).copy())
                masks.append(np.asarray(infos["action_mask"], dtype=np.int64).copy())
                values.append(vals)
                old_log_probs.append(logprobs_t)
                actions.append(action_t)
                new_observations, new_rewards, new_dones, _, new_infos = env.step(
                    action_t.cpu().numpy().astype(np.int64, copy=False) #gym runs on numpy so gotta be on cpiu
                )

                dones.append(torch.as_tensor(new_dones, device=device, dtype=torch.float32))
                rewards.append(torch.as_tensor(new_rewards, device=device, dtype=torch.float32))
                obs, infos = new_observations, new_infos

            #now combine up the experience into tensors for batch processinglater
            all_rewards = torch.stack(rewards, dim=0).transpose(0, 1)
            all_dones = torch.stack(dones, dim=0).transpose(0, 1)
            all_values = torch.stack(values, dim=0).transpose(0, 1)
            all_old_log_probs = torch.stack(old_log_probs, dim=0).transpose(0, 1)
            all_actions = torch.stack(actions, dim=0).transpose(0, 1)
            all_observations = (
                torch.as_tensor(np.stack(observations, axis=0), dtype=torch.float32)
                .transpose(0, 1)
            )
            all_masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.int64)  # (T, n, A)
            # (T, n, A) -> (n, T, A) for GAE; obs list already stacked to (n, T, d)
            masks_processed = all_masks.to(device, dtype=torch.int64).permute(1, 0, 2)
            observations_processed = all_observations.to(device, dtype=torch.float32)

            #now we need o compute last values for the last observations we saw for gae
            with torch.no_grad():
                last_o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                _, last_v = model(last_o)
            next_values = torch.zeros_like(all_values, device=device)
            next_values[:, : cfg.rollout_steps - 1] = all_values[:, 1:].clone()
            next_values[:, cfg.rollout_steps - 1] = last_v

            advantages, returns = compute_gae(
                all_rewards,
                all_dones,
                all_values,
                next_values,
                cfg.gamma,
                cfg.gae_lambda,
                device,
            )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9) #std. advantages to beinvariant to reward scale

            #now we need to create minibatches!
            batch_size = n_envs * cfg.rollout_steps
            minibatch_size = batch_size // cfg.num_minibatches

            #now reshape everything to be in correct format for minibatching
            #atp, we're mixing and matching across time and envs!
            obs_mini = observations_processed.reshape(batch_size, obs_dim)
            masks_mini = masks_processed.reshape(batch_size, num_actions)
            actions_mini = all_actions.reshape(batch_size)
            old_log_probs_mini = all_old_log_probs.reshape(batch_size)
            advantages_mini = advantages.reshape(batch_size)
            returns_mini = returns.reshape(batch_size)

            log_dict: dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
            
            #now to create minibatches, get loss, and backwards
            for _ in range(cfg.ppo_epochs):
                sample = torch.randperm(batch_size, device=device)
                for s in range(0, batch_size, minibatch_size):
                    idx = sample[s:s+minibatch_size]
                    b_obs = obs_mini.index_select(0, idx) #so along dim N*T, we select idx observations
                    b_masks = masks_mini.index_select(0, idx)
                    b_actions = actions_mini.index_select(0, idx)
                    b_old_log_probs = old_log_probs_mini.index_select(0, idx)
                    b_advantages = advantages_mini.index_select(0, idx)
                    b_returns = returns_mini.index_select(0, idx)

                    logits, new_vals = model(b_obs)
                    loss, log_dict = compute_ppo_minibatch_loss(
                        logits,
                        b_old_log_probs,
                        b_masks,
                        b_actions,
                        new_vals,
                        b_advantages,
                        b_returns,
                        cfg.clip_range,
                        cfg.value_coef,
                        cfg.entropy_coef,
                        device,
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm) #scale dradients to be not too large
                    optimizer.step()

            with torch.no_grad():
                ret_mean = all_rewards.sum(dim=1).mean().item() #for logging
            
            if (update_idx + 1) % max(cfg.log_interval, 1) == 0:
                print(
                    f"update {update_idx + 1}/{cfg.num_updates} "
                    f"mean_episode_sum_reward(rollout)={ret_mean:.4f} "
                    f"policy={log_dict['policy_loss']:.4f} "
                    f"value={log_dict['value_loss']:.4f} "
                    f"entropy={log_dict['entropy_loss']:.4f}"
                )
            if (update_idx + 1) % max(cfg.save_interval, 1) == 0:
                ck = run_dir / "checkpoints" / f"ppo_{update_idx + 1:06d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "update": update_idx + 1,
                    },
                    ck,
                )
    except Exception as e:
        raise e
    finally:
        env.close()

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_training_config(str(config_path))
    device = resolve_device(cfg.device)
    print(f"Loaded {config_path!s}  run_name={cfg.run_name!r}  device={device!s}")
    run_dir = train_ppo(cfg, device)
    print(f"Wrote run artifacts under {run_dir}")


if __name__ == "__main__":
    main()

