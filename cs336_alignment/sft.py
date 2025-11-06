import logging
import time
from typing import Callable
import os
import numpy as np
import random
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore
from einops import rearrange, einsum
import torch.distributed as dist
from tqdm import tqdm, trange
from transformers import get_cosine_schedule_with_warmup  # type: ignore
from cs336_alignment.vllm_util import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.config import load_config_from_file, SftConfig
from cs336_alignment.logger import setup_logging, print_and_log
from cs336_alignment.sft_helper import masked_normalize, get_response_log_probs, sft_microbatch_train_step


def get_batch(
    input_ids: np.ndarray,
    labels: np.ndarray,
    resp_mask: np.ndarray,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = input_ids.shape[0] - context_length
    start_idx = np.random.randint(0, sample_count, size=batch_size)

    start_idx = rearrange(start_idx, "batch -> batch 1")
    idx_range = np.arange(context_length)
    idx_range = rearrange(idx_range, "seq_len -> 1 seq_len")
    indices = start_idx + idx_range

    batch_input_ids = input_ids[indices]
    batch_labels = labels[indices]
    batch_resp_mask = resp_mask[indices]

    assert batch_input_ids.shape == (batch_size, context_length)
    assert batch_labels.shape == (batch_size, context_length)
    assert batch_resp_mask.shape == (batch_size, context_length)
    return (
        torch.from_numpy(batch_input_ids),
        torch.from_numpy(batch_labels),
        torch.from_numpy(batch_resp_mask),
    )


def cleanup():
    logging.shutdown()
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def vllm_evaluate(
    llm: PreTrainedModel,
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
):
    load_policy_into_vllm_instance(llm, vllm_model)  # # type: ignore
    evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        dump_data=False,
    )


def train_sft(
    sft_config: SftConfig,
    train_device: str,
    llm: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    resp_mask: torch.Tensor,
    eval_function: Callable[[PreTrainedModel], None] | None = None,
    print_entropy: bool = False,
):
    print_and_log("-" * 80)
    sample_count = input_ids.shape[0]
    sample_content_length = input_ids.shape[1]
    gradient_accumulation_steps = sft_config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(llm.parameters(), lr=sft_config.learning_rate)
    micro_batch_size = sft_config.micro_batch_size

    example_count = input_ids.shape[0]
    iter_batch_size = (
        sft_config.micro_batch_size * sft_config.gradient_accumulation_steps
    )
    total_training_examples = example_count * sft_config.num_epochs
    if total_training_examples < iter_batch_size:
        iter_batch_size = total_training_examples
    training_steps = total_training_examples // iter_batch_size
    print_and_log(
        f"Total training steps: {training_steps} batch size: {iter_batch_size} example count: {example_count}"
    )
    eval_interval = (
        sft_config.eval_interval * example_count // iter_batch_size
        if sft_config.eval_interval > 0
        else 0
    )
    if eval_interval > 0:
        print_and_log(f"Evaluation interval (in steps): {eval_interval}")

    start_time = time.time()
    warmup_steps = int(0.02 * training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, training_steps
    )
    for st in (pbar := trange(training_steps, desc="SFT Training Steps")):
        total_loss = torch.tensor(0.0, device=train_device)
        total_entropy = torch.tensor(0.0, device=train_device)
        for _ in trange(
            gradient_accumulation_steps, desc="Gradient Accumulation Steps", leave=False
        ):
            random_index = np.random.randint(0, sample_count, size=micro_batch_size)
            batch_input_ids, batch_labels, batch_resp_mask = (
                input_ids[random_index],
                labels[random_index],
                resp_mask[random_index],
            )
            assert batch_input_ids.shape == (micro_batch_size, sample_content_length)
            assert batch_labels.shape == (micro_batch_size, sample_content_length)
            assert batch_resp_mask.shape == (micro_batch_size, sample_content_length)
            with torch.autocast(device_type=train_device, dtype=torch.bfloat16):
                results = get_response_log_probs(
                    llm, batch_input_ids, batch_labels, return_token_entropy=print_entropy  # type: ignore
                )
                log_probs = results["log_probs"]
                if print_entropy:
                    with torch.no_grad():
                        token_entropy = results["token_entropy"]
                        assert token_entropy.shape == (micro_batch_size, sample_content_length)
                        avg_token_entropy = masked_normalize(token_entropy, batch_resp_mask, normalize_constant=micro_batch_size)
                        total_entropy += avg_token_entropy.detach()
                    
                assert log_probs.shape == (micro_batch_size, sample_content_length)
                loss, _ = sft_microbatch_train_step(
                    log_probs, batch_resp_mask, gradient_accumulation_steps, 1.0
                )
                total_loss += loss.detach()

        if sft_config.clip_gradients > 0.0:
            # implement gradient clipping if needed
            torch.nn.utils.clip_grad_norm_(
                llm.parameters(), max_norm=sft_config.clip_gradients
            )

        current_lr = lr_scheduler.get_last_lr()[0]
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss = total_loss / gradient_accumulation_steps
        total_entropy = total_entropy / gradient_accumulation_steps
        print_and_log(
            f"Step {st+1}/{training_steps} - Loss: {total_loss.item():.4f} - LR: {current_lr:.6f} - AvgEntropy: {total_entropy.item():.4f}"
        )
        pbar.set_description(f"Loss: {total_loss.item():.4f} lr: {current_lr:.6f}")  # type: ignore

        is_last_step = st == training_steps - 1
        if eval_function is not None and (
            (st + 1) % eval_interval == 0 or is_last_step
        ):
            print_and_log(f"Running evaluation at step {st+1}...")
            eval_function(llm)

    end_time = time.time()
    print_and_log(
        f"Training time for {training_steps} steps: {end_time - start_time} seconds."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="Path to config file"
    )
    parser.add_argument("-e", "--eval", action="store_true", help="Run evaluation only")
    parser.add_argument(
        "-m", "--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model ID"
    )
    args = parser.parse_args()
    config = load_config_from_file(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    setup_logging(log_file_name=f"{config_name}.log")
    print_and_log("Starting SFT training...")
    print_and_log(f"Arguments: {args}")
    sft_config = SftConfig(**config)
    if args.model_id:
        sft_config.model_id = args.model_id
    print_and_log(f"SFT Config: {sft_config}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # warning: flash-attn currently only supports certain CUDA and PyTorch versions.
    # uv add flash-attn = { url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3%2Bcu128torch2.5-cp312-cp312-linux_x86_64.whl" }
    print_and_log("Loading model and tokenizer...")
    is_eval_only = args.eval
    train_device = "cuda:0"
    llm = AutoModelForCausalLM.from_pretrained(
        f"models/{sft_config.model_id}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": train_device},
    )
    if is_eval_only:
        llm.eval()  # set model to evaluation mode
    else:
        llm.train()  # set model to training mode

    output_dir = f"models/{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    print_and_log("-" * 120)
    input_ids = np.load(f"data/input_ids_tensor.npy")[: sft_config.max_examples]
    labels = np.load(f"data/labels_tensor.npy")[: sft_config.max_examples]
    resp_mask = np.load(f"data/response_mask_tensor.npy")[: sft_config.max_examples]

    input_ids = torch.from_numpy(input_ids).to(train_device)
    resp_mask = torch.from_numpy(resp_mask).to(train_device)
    labels = torch.from_numpy(labels).to(train_device)
    print_and_log(f"Input IDs shape: {input_ids.shape}")
    assert input_ids.shape == labels.shape == resp_mask.shape

    from vllm.sampling_params import SamplingParams
    from cs336_alignment.math_baseline import (
        evaluate_vllm,
        get_evaluation_sample_params,
        get_evaluation_samples,
    )

    eval_function = None
    if sft_config.eval_interval > 0 and (torch.cuda.device_count() > 1 or is_eval_only):
        sampling_params: SamplingParams = get_evaluation_sample_params()
        prompts, ground_truths = get_evaluation_samples(256, 4096)
        print_and_log("Initializing vLLM model for evaluation...")
        vllm_model: LLM = init_vllm(
            model_id=f"models/{args.model_id}",
            device="cuda:1" if not is_eval_only else "cuda:0",
            seed=seed,
            gpu_memory_utilization=0.85,
        )
        print_and_log("Loading policy weights into vLLM model...")
        eval_function: Callable[[PreTrainedModel], None] | None = (
            lambda model: vllm_evaluate(
                model, vllm_model, prompts, ground_truths, sampling_params
            )
        )
        eval_function(llm)

    if is_eval_only:
        print_and_log("Evaluation only mode, exiting after evaluation.")
        cleanup()
        exit(0)

    tokenizer = AutoTokenizer.from_pretrained(f"models/{sft_config.model_id}")
    if sft_config.compile_model:
        print_and_log("Compiling model...")
        llm = torch.compile(llm)  # type: ignore
    train_sft(
        sft_config,
        train_device,
        llm, # type: ignore
        input_ids,
        labels,
        resp_mask,
        eval_function=eval_function,
    )
    llm.save_pretrained(save_directory=output_dir)  # type: ignore
    tokenizer.save_pretrained(save_directory=output_dir)

    cleanup()
