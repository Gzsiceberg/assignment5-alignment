import logging
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel # type: ignore
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
            dtype=torch.bfloat16, # type: ignore
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model # type: ignore
    llm_model.load_weights(state_dict.items())


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

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # warning: flash-attn currently only supports certain CUDA and PyTorch versions.
    # uv add flash-attn = { url = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3%2Bcu128torch2.5-cp312-cp312-linux_x86_64.whl" }
    print_and_log("Loading model and tokenizer...")
    train_device = "cuda:0"
    llm = AutoModelForCausalLM.from_pretrained(
        f"models/{sft_config.model_id}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": train_device},
    )
    llm.train() # set model to training mode
    tokenizer = AutoTokenizer.from_pretrained(f"models/{sft_config.model_id}")

    context_length = llm.config.max_position_embeddings
    print_and_log(f"Model context length: {context_length}")

    output_dir = f"models/sft_model_{config_name}"
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
    assert input_ids.shape[1] <= context_length, f"Input sequence length exceeds context length."

    example_count = input_ids.shape[0]
    batch_size = sft_config.micro_batch_size * sft_config.gradient_accumulation_steps
    training_steps = sft_config.num_epochs * example_count // batch_size
    print(f"Total training steps: {training_steps} batch size: {batch_size} example count: {example_count}")
    eval_interval = sft_config.eval_interval * example_count // batch_size if sft_config.eval_interval > 0 else 0
    print_and_log(f"Evaluation interval (in steps): {eval_interval}")

    from vllm.sampling_params import SamplingParams
    from math_baseline import generate_prompt_and_gt, evaluate_vllm
    vllm_model = None
    prompts = []
    ground_truths = []
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=context_length, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    from datasets import load_dataset, Dataset
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    if sft_config.eval_interval > 0 and torch.cuda.device_count() > 1:
        ds = load_dataset("hkust-nlp/dart-math-uniform")
        train: Dataset = ds["train"] # type: ignore
        print(f"Total test samples: {len(train)}")
        prompts, ground_truths = generate_prompt_and_gt(train, 512, 1024)

        print_and_log("Initializing vLLM model for evaluation...")
        vllm_model = init_vllm(
            model_id="models/Qwen/Qwen2.5-Math-1.5B",
            device="cuda:1",
            seed=seed,
            gpu_memory_utilization=0.85,
        )
        print_and_log("Loading policy weights into vLLM model...")
        load_policy_into_vllm_instance(llm, vllm_model)

        evaluate_vllm(
            vllm_model=vllm_model,
            reward_fn=lambda resp, gt: r1_zero_reward_fn(resp, gt, False),
            prompts=prompts,
            ground_truths=ground_truths,
            eval_sampling_params=sampling_params,
            dump_data=False
        )

    from sft_helper import get_response_log_probs, sft_microbatch_train_step
    import time
    from tqdm import tqdm, trange
    from transformers import get_cosine_schedule_with_warmup

    gradient_accumulation_steps = sft_config.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(llm.parameters(), lr=sft_config.learning_rate)
    batch_size = sft_config.micro_batch_size
    start_time = time.time()
    amp_ctx = torch.autocast(device_type=train_device, dtype=torch.bfloat16)
    if sft_config.compile_model:
        print_and_log("Compiling model...")
        llm = torch.compile(llm)
    sample_count = input_ids.shape[0]
    sample_content_length = input_ids.shape[1]

    warmup_steps = int(0.02 * training_steps)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps,
        training_steps
    )

    for st in (pbar := trange(training_steps, desc="SFT Training Steps")):
        total_loss = torch.tensor(0.0, device=train_device)
        for _ in trange(gradient_accumulation_steps, desc="Gradient Accumulation Steps", leave=False):
            random_index = np.random.randint(0, sample_count, size=batch_size)
            batch_input_ids, batch_labels, batch_resp_mask = input_ids[random_index], labels[random_index], resp_mask[random_index]
            assert batch_input_ids.shape == (batch_size, sample_content_length)
            assert batch_labels.shape == (batch_size, sample_content_length)
            assert batch_resp_mask.shape == (batch_size, sample_content_length)
            with amp_ctx:
                results = get_response_log_probs(
                    llm, batch_input_ids, batch_labels, return_token_entropy=False # type: ignore
                )
                log_probs = results["log_probs"]
                loss, meta_data = sft_microbatch_train_step(
                    log_probs, batch_resp_mask, gradient_accumulation_steps, 1.0
                )
                total_loss += loss.detach()

        total_loss = total_loss / gradient_accumulation_steps
        current_lr = lr_scheduler.get_last_lr()[0]
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print(f"Step {st+1}/{training_steps} - Loss: {total_loss.item():.4f} - LR: {current_lr:.6f}")
        pbar.set_description(f"Loss: {total_loss.item():.4f} lr: {current_lr:.6f}") # type: ignore

        is_last_step = st == training_steps - 1
        if vllm_model is not None and ((st + 1) % eval_interval == 0 or is_last_step):
            print_and_log(f"Running evaluation at step {st+1}...")
            load_policy_into_vllm_instance(llm, vllm_model) # # type: ignore
            evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=lambda resp, gt: r1_zero_reward_fn(resp, gt, False),
                prompts=prompts,
                ground_truths=ground_truths,
                eval_sampling_params=sampling_params,
                dump_data=False
            )

    
    llm.save_pretrained(save_directory=output_dir) # type: ignore
    tokenizer.save_pretrained(save_directory=output_dir)

    end_time = time.time()
    print_and_log(
        f"Training time for {training_steps} steps: {end_time - start_time} seconds."
    )

    logging.shutdown()
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
