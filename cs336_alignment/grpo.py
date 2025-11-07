import time
import torch
import numpy as np
from typing import Callable, Literal

from tqdm import trange
from cs336_alignment.config import RLConfig, SftConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import do_grad_accumulate
from cs336_alignment.sft_helper import masked_normalize, masked_mean
from cs336_alignment.logger import print_and_log, setup_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore


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


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, seq_len = policy_log_probs.shape
    loss, meta_info = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    masked_loss = masked_mean(loss, response_mask, dim=1, protect_zero_division=False)
    assert masked_loss.shape == (batch_size,), "Masked loss shape mismatch"
    mean_loss = masked_loss.mean() / gradient_accumulation_steps
    mean_loss.backward()
    return mean_loss, meta_info


def get_data_batch(
    micro_iter: int,
    micro_batch_size: int,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    resp_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_idx = micro_iter * micro_batch_size
    end_idx = start_idx + micro_batch_size
    assert end_idx <= input_ids.shape[0], "Batch index out of range"
    batch_input_ids = input_ids[start_idx:end_idx, :].to(input_ids.device)
    batch_labels = labels[start_idx:end_idx, :].to(labels.device)
    batch_resp_mask = resp_mask[start_idx:end_idx, :].to(resp_mask.device)
    return batch_input_ids, batch_labels, batch_resp_mask


def train_pg(
    sft_config: SftConfig,
    rl_config: RLConfig,
    train_device: str,
    llm: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    resp_mask: torch.Tensor,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
):
    print_and_log("-" * 80)
    sample_count = input_ids.shape[0]
    sample_content_length = input_ids.shape[1]
    gradient_accumulation_steps = sft_config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(
        llm.parameters(),
        lr=sft_config.learning_rate,
        fused=True,
        weight_decay=0,
        betas=(0.9, 0.95),
    )
    micro_batch_size = sft_config.micro_batch_size

    get_data_batch_fn = lambda micro_iter, micro_batch_size: get_data_batch(
        micro_iter, micro_batch_size, input_ids, labels, resp_mask
    )

    micro_batch_train_step_fn = lambda policy_log_probs, response_mask, gradient_accumulation_steps: grpo_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps,
        rl_config.loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        rl_config.cliprange,
    )

    iter_batch_size = micro_batch_size * gradient_accumulation_steps
    training_steps = sft_config.num_epochs * sample_count // iter_batch_size
    print_and_log(
        f"Total training steps: {training_steps} batch size: {iter_batch_size} example count: {sample_count}"
    )

    start_time = time.time()
    for st in (pbar := trange(training_steps, desc="PG Training Steps")):
        total_loss, total_entropy = do_grad_accumulate(
            train_device=train_device,
            llm=llm,
            get_data_batch_fn=get_data_batch_fn,
            micro_batch_train_step_fn=micro_batch_train_step_fn,
            print_entropy=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            micro_batch_size=micro_batch_size,
        )

        if sft_config.clip_gradients > 0.0:
            # implement gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(
                llm.parameters(), max_norm=sft_config.clip_gradients
            )

        optimizer.step()
        optimizer.zero_grad()

        print_and_log(
            f"Step {st+1}/{training_steps} - Loss: {total_loss.item():.4f} - Entropy: {total_entropy.item():.4f}"
        )
        pbar.set_description(f"Loss: {total_loss.item():.4f}")  # type: ignore

    end_time = time.time()
    print_and_log(
        f"Training time for {training_steps} steps: {end_time - start_time:.2f} seconds."
    )
