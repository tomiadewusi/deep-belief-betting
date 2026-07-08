from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.distributions import Categorical

from deep_belief_betting.agent_training.device import resolve_device
from deep_belief_betting.agent_training.policy.mlp_actor_critic import ActorCritic
from deep_belief_betting.agent_training.training_config import  TrainingConfig, load_training_config
from deep_belief_betting.simulation.parameters import Parameters
from deep_belief_betting.simulation.prediction_market_env import PredictionMarketEnv


ACTION_NAMES = {
    0: "HOLD",
    1: "BUY_YES",
    2: "BUY_NO",
    3: "SELL",
}


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_path: str
    training_config_path: str
    world_config_path: str
    world_config_overridden: bool
    episodes: int
    base_seed: int
    deterministic: bool
    device: str
    output_dir: str


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch scalars and arrays into JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, sort_keys=True)
        f.write("\n")


def _infer_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            "checkpoint path must be inside a run checkpoints directory, e.g. "
            "runs/ppo/<run>/checkpoints/ppo_000050.pt"
        )
    return checkpoint_path.parent.parent


def _load_required_training_config(run_dir: Path) -> Path:
    path = run_dir / "training_config.resolved.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing resolved training config: {path}")
    return path


def _load_required_world_config(run_dir: Path, override: Optional[str]) -> tuple[Path, bool]:
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"world config override not found: {path}")
        return path, True

    path = run_dir / "world_snapshot.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing world snapshot: {path}")
    return path, False


def _create_eval_dir(run_dir: Path, output_name: Optional[str]) -> Path:
    eval_root = run_dir / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    name = output_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = eval_root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask == 0, float("-inf"))


def _select_action(logits: torch.Tensor,
                   action_mask: torch.Tensor,
                   deterministic: bool) -> tuple[int, list[float]]:
    masked = _masked_logits(logits, action_mask)
    probs = torch.softmax(masked, dim=-1)

    if deterministic:
        action = int(torch.argmax(probs, dim=-1).item())
    else:
        action = int(Categorical(probs=probs).sample().item())

    probs_np = probs.detach().cpu().numpy()
    mask_np = action_mask.detach().cpu().numpy()
    probs_np = np.where(mask_np > 0, probs_np, 0.0)

    return action, probs_np.astype(float).tolist()


def _build_model(cfg: TrainingConfig,
                 env: PredictionMarketEnv, 
                 device: torch.device,
                 checkpoint_path: Path) -> ActorCritic:
    obs_dim = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    model = ActorCritic(obs_dim=obs_dim,
                        num_actions=num_actions,
                        hidden_dim=cfg.hidden_dim,
                        num_layers=cfg.num_layers,
                        device=device).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"checkpoint missing 'model' state dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _extract_step_row(episode_index: int,
                      episode_seed: int,
                      step_index: int,
                      obs: np.ndarray,
                      info_before_action: dict[str, Any],
                      action_id: int,
                      action_probs: list[float],
                      value_estimate: float,
                      reward: float,
                      terminated: bool,
                      truncated: bool,
                      info_after_action: dict[str, Any]) -> dict[str, Any]:
    
    return {"episode_index": episode_index,
            "episode_seed": episode_seed,
            "step_index": step_index,
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "action_mask": info_before_action["action_mask"].astype(int).tolist(),
            "policy_action_probabilities": action_probs,
            "value_estimate": value_estimate,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "invalid_action": bool(info_after_action.get("invalid_action", False)),
            "terminal_outcome": info_after_action.get("terminal_outcome"),
            "realised_cash_pnl": float(info_after_action.get("realised_cash_pnl", 0.0)),
            "net_pnl_if_liquidated_now": float(info_after_action.get("net_pnl_if_liquidated_now", 0.0)),
            "normalized_observation": obs.astype(float).tolist(),
            "raw_observation": dict(info_before_action.get("raw_observation", {}))}


def _summarize_episode(episode_index: int,
                       episode_seed: int,
                       rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_reward = sum(float(row["reward"]) for row in rows)
    final_row = rows[-1]

    entry_row = next(
        (row for row in rows if row["action_name"] in {"BUY_YES", "BUY_NO"}),
        None,
    )
    exit_row = next(
        (row for row in rows if row["action_name"] == "SELL"),
        None,
    )

    entry_step = entry_row["step_index"] if entry_row is not None else None
    exit_step = exit_row["step_index"] if exit_row is not None else None

    if entry_row is None:
        entry_side = None
        entry_public_probability = None
        entry_action = None
    else:
        entry_action = entry_row["action_name"]
        entry_side = "YES" if entry_action == "BUY_YES" else "NO"
        entry_public_probability = entry_row["raw_observation"].get(
            "public_probability"
        )

    exit_public_probability = (
        exit_row["raw_observation"].get("public_probability")
        if exit_row is not None
        else None
    )

    holding_duration = (
        exit_step - entry_step
        if entry_step is not None and exit_step is not None
        else None
    )

    holds_before_entry = sum(
        1
        for row in rows
        if row["action_name"] == "HOLD"
        and (entry_step is None or row["step_index"] < entry_step)
    )
    holds_while_invested = sum(
        1
        for row in rows
        if row["action_name"] == "HOLD"
        and entry_step is not None
        and row["step_index"] > entry_step
        and (exit_step is None or row["step_index"] < exit_step)
    )

    net_pnls = [float(row["net_pnl_if_liquidated_now"]) for row in rows]

    return {"episode_index": episode_index,
            "episode_seed": episode_seed,
            "final_pnl": float(final_row["realised_cash_pnl"]),
            "terminal_outcome": final_row["terminal_outcome"],
            "total_reward": float(total_reward),
            "entry_step": entry_step,
            "entry_action": entry_action,
            "entry_side": entry_side,
            "entry_public_probability": entry_public_probability,
            "exit_step": exit_step,
            "exit_public_probability": exit_public_probability,
            "holding_duration": holding_duration,
            "holds_before_entry": holds_before_entry,
            "holds_while_invested": holds_while_invested,
            "never_entered": entry_row is None,
            "entered_but_never_exited": entry_row is not None and exit_row is None,
            "max_net_pnl_if_liquidated_now": max(net_pnls) if net_pnls else None,
            "min_net_pnl_if_liquidated_now": min(net_pnls) if net_pnls else None,
            "invalid_action_count": sum(1 for row in rows if row["invalid_action"]),
            "num_steps": len(rows)}


def _mean_present(values: list[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return float(np.mean(present))


def _aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    final_pnls = np.asarray([s["final_pnl"] for s in summaries], dtype=np.float64)
    total_rewards = np.asarray([s["total_reward"] for s in summaries], dtype=np.float64)
    total_steps = sum(int(s["num_steps"]) for s in summaries)
    total_invalid = sum(int(s["invalid_action_count"]) for s in summaries)

    return {"num_episodes": len(summaries),
            "mean_final_pnl": float(np.mean(final_pnls)),
            "median_final_pnl": float(np.median(final_pnls)),
            "std_final_pnl": float(np.std(final_pnls)),
            "win_rate": float(np.mean(final_pnls > 0.0)),
            "mean_total_reward": float(np.mean(total_rewards)),
            "no_trade_rate": float(np.mean([s["never_entered"] for s in summaries])),
            "entered_but_never_exited_rate": float(
                np.mean([s["entered_but_never_exited"] for s in summaries])
            ),
            "yes_entry_count": sum(1 for s in summaries if s["entry_side"] == "YES"),
            "no_entry_count": sum(1 for s in summaries if s["entry_side"] == "NO"),
            "mean_entry_step": _mean_present([s["entry_step"] for s in summaries]),
            "mean_exit_step": _mean_present([s["exit_step"] for s in summaries]),
            "mean_holding_duration": _mean_present(
                [s["holding_duration"] for s in summaries]
            ),
            "invalid_action_rate": float(total_invalid / max(total_steps, 1))}


def _run_episode(env: PredictionMarketEnv,
                 model: ActorCritic,
                 device: torch.device,
                 episode_index: int,
                 episode_seed: int,
                 deterministic: bool,
                 max_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs, info = env.reset(seed=episode_seed)
    rows: list[dict[str, Any]] = []

    for step_index in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(info["action_mask"],
                                 dtype=torch.int64,
                                 device=device).unsqueeze(0)

        with torch.no_grad():
            logits_t, value_t = model(obs_t)
            action_id, action_probs = _select_action(logits=logits_t.squeeze(0),
                                                     action_mask=mask_t.squeeze(0),
                                                     deterministic=deterministic)

        next_obs, reward, terminated, truncated, next_info = env.step(action_id)

        row = _extract_step_row(episode_index=episode_index,
                                episode_seed=episode_seed,
                                step_index=step_index,
                                obs=obs,
                                info_before_action=info,
                                action_id=action_id,
                                action_probs=action_probs,
                                value_estimate=float(value_t.squeeze(0).item()),
                                reward=float(reward),
                                terminated=bool(terminated),
                                truncated=bool(truncated),
                                info_after_action=next_info)
        rows.append(row)

        obs = next_obs
        info = next_info

        if terminated or truncated:
            break
    else:
        raise RuntimeError(
            f"episode {episode_index} exceeded max_steps={max_steps}; "
            "environment may not be terminating"
        )

    return rows, _summarize_episode(episode_index=episode_index, 
                                    episode_seed=episode_seed,
                                    rows=rows)


def evaluate(checkpoint_path: Path,
             episodes: int,
             world_config_override: Optional[str],
             seed_override: Optional[int],
             device_override: Optional[str],
             stochastic: bool,
             output_name: Optional[str]) -> Path:

    deterministic = not stochastic
    
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    run_dir = _infer_run_dir(checkpoint_path)
    training_config_path = _load_required_training_config(run_dir)
    world_config_path, world_overridden = _load_required_world_config(run_dir,
                                                                      world_config_override)

    cfg = load_training_config(str(training_config_path))
    params = Parameters.from_yaml(world_config_path)

    base_seed = int(seed_override if seed_override is not None else params.seed)
    device_name = device_override if device_override is not None else cfg.device
    device = resolve_device(device_name)

    belief_dim = int(cfg.belief_dim) if cfg.belief_on else 0
    env = PredictionMarketEnv(params=params, belief_dim=belief_dim)

    try:
        model = _build_model(cfg=cfg,
                             env=env,
                             device=device,
                             checkpoint_path=checkpoint_path)

        output_dir = _create_eval_dir(run_dir, output_name)
        eval_config = EvalConfig(checkpoint_path=str(checkpoint_path),
                                 training_config_path=str(training_config_path),
                                 world_config_path=str(world_config_path),
                                 world_config_overridden=world_overridden,
                                 episodes=episodes,
                                 base_seed=base_seed,
                                 deterministic=deterministic,
            device=str(device),
            output_dir=str(output_dir),
        )

        with (output_dir / "eval_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(eval_config), f, sort_keys=True)

        all_step_rows: list[dict[str, Any]] = []
        episode_summaries: list[dict[str, Any]] = []

        max_steps = params.num_steps + 5

        for episode_index in range(episodes):
            episode_seed = base_seed + episode_index
            rows, summary = _run_episode(env=env,
                                         model=model,
                                         device=device,
                                         episode_index=episode_index,
                                         episode_seed=episode_seed,
                                         deterministic=not stochastic,
                                         max_steps=max_steps)
            all_step_rows.extend(rows)
            episode_summaries.append(summary)

        aggregate = _aggregate_summaries(episode_summaries)

        _write_jsonl(output_dir / "episode_steps.jsonl", all_step_rows)
        _write_jsonl(output_dir / "episode_summary.jsonl", episode_summaries)
        _write_json(output_dir / "aggregate_metrics.json", aggregate)

        print(f"Evaluation artifacts: {output_dir}")
        print(f"episodes={aggregate['num_episodes']}")
        print(f"mean_final_pnl={aggregate['mean_final_pnl']:.6f}")
        print(f"median_final_pnl={aggregate['median_final_pnl']:.6f}")
        print(f"win_rate={aggregate['win_rate']:.3f}")
        print(f"no_trade_rate={aggregate['no_trade_rate']:.3f}")
        print(
            "entered_but_never_exited_rate="
            f"{aggregate['entered_but_never_exited_rate']:.3f}"
        )
        print(f"yes_entry_count={aggregate['yes_entry_count']}")
        print(f"no_entry_count={aggregate['no_entry_count']}")
        print(f"mean_entry_step={aggregate['mean_entry_step']}")
        print(f"mean_exit_step={aggregate['mean_exit_step']}")
        print(f"invalid_action_rate={aggregate['invalid_action_rate']:.6f}")

        return output_dir

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--world-config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    evaluate(checkpoint_path=Path(args.checkpoint),
             episodes=args.episodes,
             world_config_override=args.world_config,
             seed_override=args.seed,
             device_override=args.device,
             stochastic=bool(args.stochastic),
             output_name=args.output_name)


if __name__ == "__main__":
    main()
