import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import gc
from vllm.sampling_params import SamplingParams
from vllm import LLM
from cs336_alignment.math_baseline import (
    get_evaluation_sample_params,
    get_evaluation_samples,
)
from cs336_alignment.vllm_util import init_vllm, load_policy_into_vllm_instance
from tqdm import tqdm, trange
import numpy as np
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import torch.distributed as dist
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore
import torch
import random
from cs336_alignment.sft_helper import tokenize_to_tensor
from cs336_alignment.sft import train_sft
from cs336_alignment.config import SftConfig, ExpertIterConfig, load_config_from_file
from cs336_alignment.logger import setup_logging, print_and_log


def cleanup():
    logging.shutdown()
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def expert_iter_gen(
    sample_batch_size: int,
    sampling_params: SamplingParams,
    vllm: LLM,
    sft_prompts: list[str],
    sft_responses: list[str],
    sub_batch_prompts: list[str],
    sub_batch_ground_truths: list[str],
):
    from cs336_alignment.extract import extract_ans
    from cs336_alignment.logger import print_and_log
    outputs = vllm.generate(sub_batch_prompts, sampling_params=sampling_params)
    for output, ground_truth, prompt in tqdm(
        zip(outputs, sub_batch_ground_truths, sub_batch_prompts),
        total=len(sub_batch_prompts),
        desc="Processing generated outputs",
        leave=False,
    ):
        assert len(output.outputs) == sample_batch_size
        for resp in output.outputs:
            resp_text = resp.text
            reward_dict = r1_zero_reward_fn(resp_text, ground_truth)
            if reward_dict["reward"] <= 0:
                continue
            ans = extract_ans(ground_truth, False)
            sft_prompts.append(prompt)
            sft_responses.append(f"{resp} </think> <answer> {ans} </answer>")
    
    if len(sft_prompts) == 0:
        print_and_log("Warning: No positive samples collected in this batch.")
        for i in range(4):
            prompt = sub_batch_prompts[i]
            ground_truth = sub_batch_ground_truths[i]
            sft_prompts.append(prompt)
            ans = extract_ans(ground_truth, False)
            sft_responses.append(f"{ground_truth} </think> <answer> {ans} </answer>")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="Path to config file"
    )
    parser.add_argument(
        "-t", "--test", action="store_true", help="Run in test mode"
    )
    args = parser.parse_args()
    config = load_config_from_file(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    setup_logging(log_file_name=f"{config_name}.log")
    print_and_log("Starting SFT training...")
    print_and_log(f"Arguments: {args}")
    sft_config = SftConfig(**config.get("SftConfig", {}))
    expert_iter_config = ExpertIterConfig(**config.get("ExpertIterConfig", {}))
    print_and_log(f"SFT Config: {sft_config}")
    print_and_log(f"Expert Iter Config: {expert_iter_config}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_id = sft_config.model_id
    question_batch_size = expert_iter_config.question_batch_size
    vllm_batch_size = 64

    gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    if gpu_memory >= 64:
        vllm_batch_size = int(64 / 8 * vllm_batch_size)
    elif gpu_memory >= 32:
        vllm_batch_size = int(32 / 8 * vllm_batch_size)
    elif gpu_memory >= 16:
        vllm_batch_size = int(16 / 8 * vllm_batch_size)
    vllm_batch_size = min(vllm_batch_size, question_batch_size)
    print_and_log(f"Using vLLM batch size: {vllm_batch_size}")

    sample_batch_size = expert_iter_config.sample_batch_size
    n_ei_steps = expert_iter_config.n_ei_steps
    base_name = os.path.splitext(os.path.basename(args.config))[0]
    output_model = base_name

    prompts, ground_truths = get_evaluation_samples(sft_config.max_examples, 0)
    sampling_params = get_evaluation_sample_params(sample_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_id}")

    gpus_count = torch.cuda.device_count()
    vllm_device = "cuda:0" if gpus_count == 1 else "cuda:1"
    vllm = None
    train_device = "cuda:0"
    llm: PreTrainedModel | None = None
    indices = np.array(range(len(prompts)))
    is_sample_device = vllm_device == train_device
    output_dir = f"models/{output_model}"
    os.makedirs(output_dir, exist_ok=True)
    for _ in range(n_ei_steps):
        np.random.shuffle(indices)
        batch_indices = indices[:question_batch_size]
        batch_prompts = [prompts[j] for j in batch_indices]
        batch_ground_truths = [ground_truths[j] for j in batch_indices]
        assert len(batch_prompts) == question_batch_size
        assert len(batch_ground_truths) == question_batch_size

        if vllm is None:
            print("Initializing vLLM model for EI step...")
            vllm = init_vllm(
                model_id=f"models/{model_id}",
                device=vllm_device,
                seed=42,
                gpu_memory_utilization=0.85,
            )
            if llm is not None:
                load_policy_into_vllm_instance(llm, vllm)

        sft_prompts = []
        sft_responses = []

        for i in trange(
            0, question_batch_size, vllm_batch_size, desc="Generating batches"
        ):
            sub_batch_prompts = batch_prompts[i : i + vllm_batch_size]
            sub_batch_ground_truths = batch_ground_truths[i : i + vllm_batch_size]
            expert_iter_gen(
                sample_batch_size,
                sampling_params,
                vllm,
                sft_prompts,
                sft_responses,
                sub_batch_prompts,
                sub_batch_ground_truths,
            )
        assert len(sft_prompts) > 0, "No positive samples collected in this EI step."
        print(f"Number of positive samples collected: {len(sft_prompts)}")

        # Free up vLLM memory if on the same device
        if is_sample_device:
            del vllm
            vllm = None
            print_and_log("Clearing vLLM from memory...")
            gc.collect()

        tokenized_data = tokenize_to_tensor(sft_prompts, sft_responses, tokenizer)
        input_ids = tokenized_data["input_ids"].to(train_device)
        response_mask = tokenized_data["response_mask"].to(train_device)
        labels = tokenized_data["labels"].to(train_device)
        print_and_log("Tokenization complete.")
        print_and_log(f"Input IDs shape: {input_ids.shape}")

        if args.test:
            break

        if llm is None:
            print_and_log("Loading model for SFT training...")
            llm = AutoModelForCausalLM.from_pretrained(
                f"models/{sft_config.model_id}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": train_device},
            )
            llm.train()  # type: ignore

        train_sft(
            sft_config,
            train_device,
            llm=llm, # type: ignore
            input_ids=input_ids,
            labels=labels,
            resp_mask=response_mask,
            print_entropy=True,
        )
    
    if llm is not None:
        print_and_log(f"Saving fine-tuned model to {output_dir}...")
        llm.save_pretrained(save_directory=output_dir)  # type: ignore
        tokenizer.save_pretrained(save_directory=output_dir)
    cleanup()
