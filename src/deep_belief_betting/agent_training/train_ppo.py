from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from model.model import Architecture3

from deep_belief_betting.env_factory import make_vector_env
from deep_belief_betting.parameters import Parameters
from deep_belief_betting.agent_training.algorithms.ppo import compute_gae, compute_ppo_minibatch_loss, masked_log_prob_and_entropy

from deep_belief_betting.agent_training.device import resolve_device
from deep_belief_betting.agent_training.policy.mlp_actor_critic import ActorCritic
from deep_belief_betting.agent_training.run_layout import create_run_dir
from deep_belief_betting.agent_training.training_config import TrainingConfig, load_training_config

import json

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

def log_dict_pretty(log_dict: dict[str, float]) -> str:
    rounded = {k: round(v, 5) for k, v in log_dict.items()}
    return json.dumps(rounded, sort_keys=True, separators=(", ", ": "))

def log_dict_to_tensorboard(writer: SummaryWriter, step: int, log_dict: dict[str, float]) -> None:
    for key, value in log_dict.items():
        writer.add_scalar(f"metrics/{key}", value, step)
    writer.flush()

def log_writer(run_dir: Path, use_tb: bool) -> Optional[Any]:
    if not use_tb:
        return None
    tb = run_dir / "tb"
    tb.mkdir(exist_ok=True, parents=True)
    return SummaryWriter(log_dir=str(tb))

def train_ppo(cfg: TrainingConfig, device: torch.device) -> Path:
    set_seed(cfg.seed)
    run_dir = create_run_dir(cfg.log_dir, cfg.run_name, cfg)
    writer = log_writer(run_dir, cfg.use_tensorboard)
    if writer is not None:
        print(f"TensorBoard: tensorboard --logdir ./{run_dir!s}/tb")

    params = Parameters.from_yaml(cfg.world_yaml_base_path)
    belief_dim = int(cfg.belief_dim) if cfg.belief_on else 0
    try:
        env = make_vector_env(params, cfg.num_envs, belief_dim=belief_dim)
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

        ### BELIEF MODEL SETUP ###
        belief_model = None
        belief_model_cfg = None
        seq_bufs: Optional[list[list]] = None
        if cfg.belief_on and cfg.belief_checkpoint_path:
            ck = torch.load(cfg.belief_checkpoint_path, map_location=device, weights_only=False)
            belief_model_cfg = SimpleNamespace(**ck["cfg"])
            belief_model = Architecture3(belief_model_cfg)
            belief_model.load_state_dict(ck["model"])
            belief_model.eval()
            belief_model.to(device)
            if cfg.belief_dim != belief_model_cfg.d_z:
                raise ValueError(
                    f"belief_dim in training config ({cfg.belief_dim}) does not match "
                    f"d_z in belief model checkpoint ({belief_model_cfg.d_z}). "
                    f"Set belief_dim: {belief_model_cfg.d_z} in ppo_train.yaml."
                )
            if not params.belief_features.enabled or params.belief_features.mode != "vector":
                print("[belief] WARNING: belief_on=true but world config has belief_features.enabled=false "
                      "or mode!=vector — z_t will not appear in observations.")
            if cfg.belief_q_on and not params.features.include_belief_q:
                print("[belief] WARNING: belief_q_on=true but world config has include_belief_q=false "
                      "— Q will not appear in observations.")
            seq_bufs = [[] for _ in range(n_envs)]
            print(f"[belief] loaded Architecture3  d_z={belief_model_cfg.d_z}  path={cfg.belief_checkpoint_path}")
        ### END BELIEF MODEL SETUP ###

        obs, infos = env.reset(seed=cfg.seed)

        #number of policy updates
        for update_idx in range(cfg.num_updates):
            rewards, dones, values, old_log_probs, actions, masks, observations, pnl_steps = [], [], [], [], [], [], [], []

            #actual policy rollout for experience collection - happens in parallel across envs
            for i in range(cfg.rollout_steps):
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                mask_t = torch.as_tensor(np.array(list(infos["action_mask"]), dtype=np.int64), device=device)

                logits, vals = model(obs_t)
                with torch.no_grad():
                    action_t, logprobs_t = _masked_action_sample(logits, mask_t)

                observations.append(np.asarray(obs, dtype=np.float32).copy())
                masks.append(np.array(list(infos["action_mask"]), dtype=np.int64).copy())
                values.append(vals)
                old_log_probs.append(logprobs_t)
                actions.append(action_t)
                new_observations, new_rewards, new_dones, _, new_infos = env.step(
                    action_t.cpu().numpy().astype(np.int64, copy=False) #gym runs on numpy so gotta be on cpiu
                )

                dones.append(torch.as_tensor(new_dones, device=device, dtype=torch.float32))
                rewards.append(torch.as_tensor(new_rewards, device=device, dtype=torch.float32))
                obs, infos = new_observations, new_infos
                pnl = np.where(new_dones > 0, new_infos["realised_cash_pnl"], 0)
                pnl_steps.append(pnl)

                ### BELIEF MODEL INFERENCE ###
                if belief_model is not None:
                    max_seq = belief_model_cfg.T + 1
                    for env_i, e in enumerate(env.envs):
                        if new_dones[env_i]:
                            seq_bufs[env_i] = []  # episode ended — clear history
                        else:
                            ms = e.unwrapped.market.get_state()
                            seq_bufs[env_i].append([ms.public_probability, ms.delta_q])
                            if len(seq_bufs[env_i]) > max_seq:
                                seq_bufs[env_i] = seq_bufs[env_i][-max_seq:]

                    with torch.no_grad():
                        for env_i, e in enumerate(env.envs):
                            if not seq_bufs[env_i]:
                                continue
                            x = torch.tensor(seq_bufs[env_i], dtype=torch.float32, device=device).unsqueeze(0)
                            p_t, _, z_t = belief_model(x)
                            base = e.unwrapped
                            base.set_belief_vector(z_t[0].cpu().numpy())
                            if cfg.belief_q_on:
                                base.set_belief_q(float(p_t[0]))
                ### END BELIEF MODEL INFERENCE ###

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
            final_logger: dict[str, float] = {
                "policy_loss": 0.0, 
                "value_loss": 0.0, 
                "entropy_loss": 0.0, 
                "total_loss": 0.0,
                "denominator_for_avg": 0,
                "kl_div": 0.0,
                "grad_norm_total": 0.0, #this one specifically to see if the gradient clipping is actually helpful
                "pnl_avg": 0.0
            }
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
                    new_log_prob, _ = masked_log_prob_and_entropy(
                        logits, b_masks, b_actions, device
                    )
                    final_logger["kl_div"] += float((b_old_log_probs - new_log_prob.detach()).mean())

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
                    final_logger["policy_loss"] += log_dict["policy_loss"]
                    final_logger["value_loss"] += log_dict["value_loss"]
                    final_logger["entropy_loss"] += log_dict["entropy_loss"]
                    final_logger["total_loss"] += float(loss)
                    final_logger["denominator_for_avg"] += 1
                    optimizer.zero_grad()
                    loss.backward()
                    gn = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)) #clip dradients to be not too large
                    final_logger["grad_norm_total"] += gn
                    optimizer.step()

            final_logger["policy_loss"] /= final_logger["denominator_for_avg"]
            final_logger["value_loss"] /= final_logger["denominator_for_avg"]
            final_logger["entropy_loss"] /= final_logger["denominator_for_avg"]
            final_logger["total_loss"] /= final_logger["denominator_for_avg"]
            final_logger["kl_div"] /= final_logger["denominator_for_avg"]
            final_logger["grad_norm_total"] /= final_logger["denominator_for_avg"]

            with torch.no_grad():
                rew_mean = all_rewards.sum(dim=1).mean().item() #for logging

            pnl_stack = np.stack(pnl_steps, axis=0)  # (T, n_envs)
            dones = all_dones.T.float().cpu().numpy()  # (T, n_envs), all_dones is (n_envs, T)
            
            #only want to keep pnl from when the episode is done, cuz then we have the actual pnl
            pnl_resolved = pnl_stack[dones > 0]
            final_logger["pnl_avg"] = pnl_resolved.mean()
            step = update_idx

            if writer is not None:
                log_dict_to_tensorboard(writer, step, final_logger)

            if (update_idx + 1) % max(cfg.log_interval, 1) == 0:
                print(log_dict_pretty(final_logger))
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
        if writer is not None:
            writer.close()
        if env is not None:
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