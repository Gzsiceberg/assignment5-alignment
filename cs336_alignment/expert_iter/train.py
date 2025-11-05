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
    from rich import print
    from cs336_alignment.sft_helper import tokenize_to_tensor
    from cs336_alignment.sft import train_sft
    from cs336_alignment.config import SftConfig, ExpertIterConfig

    sft_config = SftConfig()
    expert_iter_config = ExpertIterConfig()

    model_id = sft_config.model_id
    question_batch_size = expert_iter_config.question_batch_size
    vllm_batch_size = expert_iter_config.vllm_batch_size
    sample_batch_size = expert_iter_config.sample_batch_size
    n_ei_steps = expert_iter_config.n_ei_steps
    output_model = expert_iter_config.output_model_dir

    prompts, ground_truths = get_evaluation_samples(sft_config.max_examples, 0)
    sampling_params = get_evaluation_sample_params(sample_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_id}")

    gpus_count = torch.cuda.device_count()
    vllm_device = "cuda:0" if gpus_count == 1 else "cuda:1"
    vllm = init_vllm(
        model_id=f"models/{model_id}",
        device=vllm_device,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    train_device = "cuda:0"
    llm = AutoModelForCausalLM.from_pretrained(
        f"models/{sft_config.model_id}",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": train_device},
    )
    llm.train()  # set model to training mode

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

        train_sft(
            sft_config,
            train_device,
            llm=llm,
            input_ids=input_ids,
            labels=labels,
            resp_mask=response_mask,
        )
    
    llm.save_pretrained(save_directory=output_dir)  # type: ignore
    tokenizer.save_pretrained(save_directory=output_dir)
    cleanup()
