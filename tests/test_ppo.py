import torch

from deep_belief_betting.agent_training.algorithms.ppo import (
    compute_gae,
    masked_log_prob_and_entropy,
)


def test_masked_log_prob_excludes_invalid_actions() -> None:
    logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 0, 1, 0]])
    actions = torch.tensor([0])

    log_probs, entropy = masked_log_prob_and_entropy(
        logits=logits,
        action_mask=mask,
        actions=actions,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    assert log_probs.item() < 0.0


def test_compute_gae_resets_at_done_boundary() -> None:
    rewards = torch.tensor([[1.0, 1.0, 1.0]])
    dones = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.full_like(rewards, 10.0)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_values=next_values,
        gamma=0.9,
        gae_lambda=1.0,
        device=torch.device("cpu"),
    )

    assert advantages[0, 1].item() == 1.0
    assert returns[0, 1].item() == 1.0
