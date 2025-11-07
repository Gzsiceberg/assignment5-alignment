import gc
import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import time
import torch
import numpy as np
from typing import Callable, Literal

from tqdm import tqdm, trange
from vllm import LLM, SamplingParams
from cs336_alignment.config import RLConfig, SftConfig
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import get_evaluation_sample_params, get_evaluation_samples
from cs336_alignment.sft import cleanup, do_grad_accumulate, vllm_evaluate
from cs336_alignment.sft_helper import masked_normalize, masked_mean, tokenize_to_tensor
from cs336_alignment.logger import print_and_log, setup_logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel # type: ignore
from cs336_alignment.vllm_util import init_vllm  # type: ignore
from cs336_alignment.config import load_config_from_file
import typer


def repeat_items_by_group_size(items: list, group_size: int) -> list:
    """Efficiently repeat each item in the list by group_size times.
    
    Args:
        items: List of items to repeat
        group_size: Number of times to repeat each item
    
    Returns:
        List with each item repeated group_size times in sequence
    """
    return [item for item in items for _ in range(group_size)]


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


def rollout(sampling_params: SamplingParams, vllm: LLM, prompts: list[str]) -> list[str]:
    n = sampling_params.n
    outputs = vllm.generate(prompts, sampling_params=sampling_params)
    rollout_responses = []
    for output in tqdm(outputs, total=len(prompts), desc="Processing generated outputs", leave=False):
        assert len(output.outputs) == n, "Output count mismatch in rollout"
        for resp in output.outputs:
            resp_text = resp.text
            rollout_responses.append(resp_text)
    return rollout_responses


def train(config_name: str = typer.Argument("config/grpo_test.yaml")):
    config = load_config_from_file(config_name)
    sft_config: SftConfig = SftConfig(**config["SftConfig"])
    rl_config: RLConfig = RLConfig(**config["RLConfig"])
    base_name = os.path.splitext(os.path.basename(config_name))[0]
    setup_logging(f"{base_name}.log")

    model_id = sft_config.model_id
    sample_question_size = rl_config.rollout_batch_size // rl_config.group_size
    assert rl_config.rollout_batch_size % rl_config.group_size == 0, "Rollout batch size must be divisible by group size"
    if rl_config.rollout_batch_size != sft_config.micro_batch_size * sft_config.gradient_accumulation_steps:
        print_and_log(
            f"Warning: Rollout batch size {rl_config.rollout_batch_size} does not equal micro_batch_size * gradient_accumulation_steps ({sft_config.micro_batch_size} * {sft_config.gradient_accumulation_steps} = {sft_config.micro_batch_size * sft_config.gradient_accumulation_steps}). This may lead to inefficient training."
        )
        sft_config.micro_batch_size = rl_config.rollout_batch_size // sft_config.gradient_accumulation_steps
    assert rl_config.rollout_batch_size == sft_config.micro_batch_size * sft_config.gradient_accumulation_steps, \
        "Rollout batch size must equal micro_batch_size * gradient_accumulation_steps"
    
    print_and_log(f"sft_config: {sft_config}")
    print_and_log(f"rl_config: {rl_config}")

    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_id}")

    prompts, ground_truths = get_evaluation_samples(sft_config.max_examples, 0)
    sampling_params = get_evaluation_sample_params(rl_config.group_size, 2048 - 512 - 256)

    eval_prompts, eval_ground_truths = get_evaluation_samples(512, 4096)
    eval_sampling_params: SamplingParams = get_evaluation_sample_params(1, 2048 - 512 - 256)

    output_model = base_name
    gpus_count = torch.cuda.device_count()
    vllm_device = "cuda:0" if gpus_count == 1 else "cuda:1"
    vllm = None
    train_device = "cuda:0"
    llm: PreTrainedModel | None = None
    question_ids = np.array(range(len(prompts)))
    is_sample_device = vllm_device == train_device
    output_dir = f"models/{output_model}"

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    rollout_batch_size = rl_config.rollout_batch_size
    for step in (pbar:= trange(rl_config.steps, desc="GRPO Overall Steps")):
        is_last_step = (step + 1) == rl_config.steps
        print_and_log(f"GRPO Overall Step {step+1}/{rl_config.steps} starting...")
        sample_question_ids = np.random.choice(
            question_ids,
            size=sample_question_size,
            replace=False,
        )
        sample_question_ids = list(sample_question_ids)
        assert len(sample_question_ids) == sample_question_size
        sample_prompts = [prompts[j] for j in sample_question_ids]
        assert len(sample_prompts) == sample_question_size
        sample_ground_truths = [ground_truths[j] for j in sample_question_ids]
        assert len(sample_ground_truths) == sample_question_size

        if vllm is None:
            print("Initializing vLLM model for grpo step...")
            vllm = init_vllm(
                model_id=f"models/{model_id}",
                device=vllm_device,
                seed=42,
                gpu_memory_utilization=0.85,
            )

        if llm is not None and ((step + 1) % 10 == 0 or is_last_step):
            vllm_evaluate(
                llm, vllm, eval_prompts, eval_ground_truths, eval_sampling_params
            )
        
        # Create rollout lists efficiently using helper function
        rollout_prompts = repeat_items_by_group_size(sample_prompts, rl_config.group_size)
        rollout_ground_truths = repeat_items_by_group_size(sample_ground_truths, rl_config.group_size)
        assert len(rollout_ground_truths) == rollout_batch_size, f"Expected {rollout_batch_size} rollout ground truths, got {len(rollout_ground_truths)}"
        assert len(rollout_prompts) == rollout_batch_size, f"Expected {rollout_batch_size} rollout prompts, got {len(rollout_prompts)}"

        rollout_responses = rollout(sampling_params, vllm, sample_prompts)
        assert len(rollout_responses) == rollout_batch_size, f"Expected {rollout_batch_size} rollout responses, got {len(rollout_responses)}"

        advantages, raw_rewards, reward_meta_info = compute_group_normalized_rewards(
            rollout_responses,
            rollout_ground_truths,
            rl_config.group_size,
            rl_config.advantage_eps,
            rl_config.use_std_normalization,
        )
        advantages = advantages.to(train_device)
        raw_rewards = raw_rewards.to(train_device)
        mean_reward = reward_meta_info["mean_reward"]
        print_and_log(f"Mean Reward for this rollout: {mean_reward:.4f}")

        # Free up vLLM memory if on the same device
        if is_sample_device:
            del vllm
            vllm = None
            print_and_log("Clearing vLLM from memory...")
            gc.collect()
        
        tokenized_data = tokenize_to_tensor(rollout_prompts, rollout_ground_truths, tokenizer)
        input_ids = tokenized_data["input_ids"].to(train_device)
        response_mask = tokenized_data["response_mask"].to(train_device)
        labels = tokenized_data["labels"].to(train_device)
        print_and_log("Tokenization complete.")
        print_and_log(f"Input IDs shape: {input_ids.shape}")


        if llm is None:
            print_and_log("Loading model for SFT training...")
            llm = AutoModelForCausalLM.from_pretrained(
                f"models/{sft_config.model_id}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": train_device},
            )
            llm.train()  # type: ignore

        train_pg(
            sft_config=sft_config,
            rl_config=rl_config,
            train_device=train_device,
            llm=llm,  # type: ignore
            input_ids=input_ids,
            labels=labels,
            resp_mask=response_mask,
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=None,
        )

        if llm is not None and ((step + 1) % 10 == 0 or is_last_step):
            print_and_log(f"Saving model checkpoint at step {step+1}...")
            llm.save_pretrained(output_dir)


if __name__ == "__main__":
    try:
        typer.run(train)
    finally:
        cleanup()