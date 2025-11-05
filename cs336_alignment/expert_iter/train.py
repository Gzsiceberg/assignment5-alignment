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
import os
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
    sft_respones: list[str],
    sub_batch_prompts: list[str],
    sub_batch_ground_truths: list[str],
):
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
            if reward_dict["reward"] > 0:
                sft_prompts.append(prompt)
                sft_respones.append(resp_text)


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
    print_and_log(f"Using vLLM batch size: {vllm_batch_size}")

    sample_batch_size = expert_iter_config.sample_batch_size
    n_ei_steps = expert_iter_config.n_ei_steps
    output_model = expert_iter_config.output_model_dir

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
        print(f"Number of positive samples collected: {len(sft_prompts)}")

        # Free up vLLM memory if on the same device
        if is_sample_device:
            del vllm
            vllm = None
            gc.collect()

        tokenized_data = tokenize_to_tensor(sft_prompts, sft_responses, tokenizer)
        input_ids = tokenized_data["input_ids"]
        response_mask = tokenized_data["response_mask"]
        labels = tokenized_data["labels"] 

        if args.test:
            break

        if llm is None:
            llm = AutoModelForCausalLM.from_pretrained(
                f"models/{sft_config.model_id}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": train_device},
            )
            llm.train()  # type: ignore

        print_and_log("Starting SFT training step...")
        train_sft(
            sft_config,
            train_device,
            llm=llm, # type: ignore
            input_ids=input_ids,
            labels=labels,
            resp_mask=response_mask,
        )
    
    if llm is not None:
        print_and_log(f"Saving fine-tuned model to {output_dir}...")
        llm.save_pretrained(save_directory=output_dir)  # type: ignore
        tokenizer.save_pretrained(save_directory=output_dir)
    cleanup()
