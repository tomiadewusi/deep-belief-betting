from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_log_prob_and_entropy(logits: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = logits.masked_fill(action_mask == 0, -float('inf')).to(device)
    dist = torch.distributions.Categorical(logits=masked_logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_probs, entropy

def compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, gamma: float, gae_lambda: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Given experience, compute the Generalized Advantage Estimation (GAE) advantages.
    Formula: 
    TD error (t) = reward_t + (1 - done_t) * (gamma * Value(s_{t+1})) - Value(s_t)
    Advantage(t) = sum_{i=0}^{T-1-t} (gamma * gae_lambda) ^ i * TD error (t+i)
    Advantage(t) semantically: A(s_t, a_t) = Q(s_t, a_t) - V(s_t) a.k.a how much better is taking action a_t at state s_t compared to the value i think i would've gotten under my old policy?

    Here, GAE lambda is saying how much do I trust value function compared to real experience? High gae_lambda means more trust in real experience, low means trust the value func.

    We use advantage to compute returns: Returns(t) = Advantage(t) + Value(s_t)
    Advantages are used for policy loss, values are used for critic loss.

    Shapes:
    rewards = (N, T)

    '''
    advantages = torch.zeros_like(rewards).to(device)
    dones = dones.float().to(device)

    td_errors = rewards + (1 - dones) * gamma * next_values - values
    
    T = rewards.shape[1]
    N = rewards.shape[0]

    gae = torch.zeros(N).to(device)
    
    for t in reversed(range(T)):
        gae = td_errors[:, t] + gamma * gae_lambda * (1-dones[:, t]) * gae
        advantages[:, t] = gae
    
    returns = advantages + values

    return advantages, returns


def compute_ppo_minibatch_loss(
    logits: torch.Tensor,
    old_log_probs: torch.Tensor, 
    action_mask: torch.Tensor, 
    actions: torch.Tensor,
    new_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    '''
    PPO has three losses:
    1) Policy loss: compute ratio b/w prob of action given new policy / prob of action given old policy for a given state. Then scale by advantage, then clip.
    2) Value loss: MSE(Value(s_t), Returns(t))
    3) Entropy loss: to encourage exploration, encourage higher entropy (distributions more spread out), so take mean entropy across all timesteps.

    return loss per episode, dict containing loss values for each loss type for logging
    '''
    log_probs, entropy = masked_log_prob_and_entropy(logits, action_mask, actions, device)

    ratio = (log_probs - old_log_probs).exp()
    clip_ratio = ratio.clamp(1-clip_range, 1+clip_range)
    policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
    value_loss = F.mse_loss(new_values, returns)
    entropy_loss = entropy.mean()

    return policy_loss + value_coef * value_loss - entropy_coef * entropy_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
    }