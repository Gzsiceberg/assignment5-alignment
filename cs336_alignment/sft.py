import logging
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
import torch
import numpy as np
from einops import rearrange, einsum



def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=str(torch.bfloat16),
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def get_batch(
    input_ids: np.ndarray,
    labels: np.ndarray,
    resp_mask: np.ndarray,
    batch_size: int,
    context_length: int,
    limit_samples: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_count = input_ids.shape[0] - context_length
    if limit_samples > 0:
        sample_count = min(limit_samples, sample_count)
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


if __name__ == "__main__":
    import os
    import argparse
    import numpy as np
    import random
    from cs336_alignment.config import load_config_from_file, SftConfig
    from cs336_alignment.logger import setup_logging, print_and_log
    from rich import print

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="Path to config file"
    )
    args = parser.parse_args()
    config = load_config_from_file(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    setup_logging(log_file_name=f"{config_name}.log")
    print_and_log("Starting SFT training...")
    print_and_log(f"Arguments: {args}")
    sft_config = SftConfig(**config)
    print_and_log(f"SFT Config: {sft_config}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_device = "cuda:0"
    llm = AutoModelForCausalLM.from_pretrained(
        "models/Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen2.5-Math-1.5B")

    context_length = llm.config.max_position_embeddings
    print_and_log(f"Model context length: {context_length}")

    output_dir = f"models/sft_model_{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    print_and_log("-" * 120)
    input_ids = np.memmap(f"data/input_ids_train.npy", mode="r", dtype=np.int32)
    labels = np.memmap(f"data/labels_train.npy", mode="r", dtype=np.int32)
    resp_mask = np.memmap(f"data/response_mask_train.npy", mode="r", dtype=bool)
    print_and_log(f"Training data has {input_ids.shape[0] / 1_000_000:.2f}M tokens.")

    from sft_helper import get_response_log_probs, sft_microbatch_train_step
    import time
    from tqdm import tqdm, trange

    gradient_accumulation_steps = sft_config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(llm.parameters(), lr=sft_config.learning_rate)
    batch_size = sft_config.batch_size
    start_time = time.time()

    pbar = trange(sft_config.num_epochs, desc="SFT Epoch")
    for epoch in pbar:
        batch_input_ids, batch_labels, batch_resp_mask = get_batch(
            input_ids, labels, resp_mask, batch_size, context_length, sft_config.limit
        )
        batch_input_ids = batch_input_ids.to(train_device)
        batch_labels = batch_labels.to(train_device)
        batch_resp_mask = batch_resp_mask.to(train_device)

        results = get_response_log_probs(
            llm, batch_input_ids, batch_labels, return_token_entropy=True
        )
        log_probs = results["log_probs"]
        token_entropy = results["token_entropy"]
        loss, meta_data = sft_microbatch_train_step(
            log_probs, batch_resp_mask, gradient_accumulation_steps, 1.0
        )

        if (epoch + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"SFT Epoch {epoch+1} | Loss: {loss.item():.4f}")
    
    llm.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)

    end_time = time.time()
    print_and_log(
        f"Training time for {sft_config.num_epochs} epochs: {end_time - start_time} seconds."
    )
    logging.shutdown()
    del input_ids
    del labels
    del resp_mask
