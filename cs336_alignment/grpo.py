import torch
import numpy as np
from typing import Callable, Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def compute_group_normalized_rewards(
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if reward_fn is None:
        reward_fn = lambda ans, gt: r1_zero_reward_fn(ans, gt, fast=False)

    prompt_size = len(rollout_responses) // group_size
    assert (
        len(rollout_responses) == prompt_size * group_size
    ), "Mismatch in total responses and group size"
    rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        assert resp is not None, "Response should not be None"
        assert gt is not None, "Ground truth should not be None"
        reward_dict = reward_fn(resp, gt)
        rewards.append(reward_dict["reward"])
    rewards = torch.tensor(rewards, dtype=torch.float32).view(prompt_size, group_size)
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - mean_rewards
    if normalize_by_std:
        std_dev = advantages.std(dim=1, keepdim=True) + advantage_eps
        advantages = advantages / std_dev
    return (
        advantages.view(-1),
        rewards.view(-1),
        {"mean_reward": mean_rewards.mean().item()},
    )


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -policy_log_probs * raw_rewards_or_advantages


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    assert advantages.shape == (batch_size, 1), "Advantages shape mismatch"
    ratio = torch.exp(policy_log_probs - old_log_probs)
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    normal_loss = ratio * advantages
    clip_loss = ratio_clipped * advantages
    loss = -torch.min(normal_loss, clip_loss)
    assert loss.shape == (batch_size, seq_len), "Loss shape mismatch"
    clip_count = torch.abs(loss - clip_loss) < 1e-5
    meta_info = {"clip_fraction": clip_count.float().mean()}
    return loss, meta_info


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    meta_info = {}
    match loss_type:
        case "no_baseline":
            assert raw_rewards is not None, "Raw rewards required for no_baseline loss"
            loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        case "reinforce_with_baseline":
            assert (
                advantages is not None
            ), "Advantages required for reinforce_with_baseline loss"
            loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        case "grpo_clip":
            assert advantages is not None, "Advantages required for grpo_clip loss"
            assert (
                old_log_probs is not None
            ), "Old log probs required for grpo_clip loss"
            assert cliprange is not None, "Cliprange required for grpo_clip loss"
            loss, meta_info_clip = compute_grpo_clip_loss(
                advantages, policy_log_probs, old_log_probs, cliprange
            )
            meta_info.update(meta_info_clip)

    assert loss.shape == (batch_size, seq_len), "Final loss shape mismatch"
    return loss, meta_info
