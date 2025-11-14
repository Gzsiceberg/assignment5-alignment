from vllm import LLM
from transformers import PreTrainedModel  # type: ignore
from unittest.mock import patch
import torch


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
) -> LLM:
    from vllm.model_executor import set_random_seed as vllm_set_random_seed

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
            dtype=torch.bfloat16,  # type: ignore
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model  # type: ignore
    llm_model.load_weights(state_dict.items())


if __name__ == "__main__":
    from cs336_alignment.logger import print_and_log
    from cs336_alignment.math_baseline import (
        get_evaluation_sample_params,
        get_evaluation_samples,
        evaluate_vllm,
    )
    import argparse
    import numpy as np
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--offset", type=int, default=4096)
    parser.add_argument(
        "-m", "--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model ID"
    )
    args = parser.parse_args()

    seed = args.seed
    limit = args.limit
    offset = args.offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    prompts, ground_truths = get_evaluation_samples(limit, offset)
    sampling_params = get_evaluation_sample_params()

    print_and_log("Initializing vLLM model for evaluation...")
    vllm_model = init_vllm(
        model_id=f"models/{args.model_id}",
        device="cuda:0",
        seed=seed,
        gpu_memory_utilization=0.85,
    )

    evaluate_vllm(
        vllm_model=vllm_model,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        dump_data=False,
    )

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
