from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
import torch


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
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    resp_mask: torch.Tensor,
    batch_size: int,
):
    import numpy as np

    sample_count = input_ids.shape[0]
    indices = np.random.randint(0, sample_count, size=batch_size)
    batch_input_ids = input_ids[indices]
    batch_labels = labels[indices]
    batch_resp_mask = resp_mask[indices]
    return batch_input_ids, batch_labels, batch_resp_mask


if __name__ == "__main__":
    train_device = "cuda:0"
    llm = AutoModelForCausalLM.from_pretrained(
        "models/Qwen/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        default_device=train_device,
    )
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen/Qwen2.5-Math-1.5B")

    import datasets
    from datasets import load_dataset
    from sft_helper import extract_prompt_and_response, tokenize_prompt_and_output
    import os

    ds = load_dataset("hkust-nlp/dart-math-uniform")
    train: datasets.Dataset = ds["train"]  # type: ignore

    output_dir = "models/sft"
    os.makedirs(output_dir, exist_ok=True)

    prompts, responses = extract_prompt_and_response(train)
    tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = tokenized_data["input_ids"].to(train_device)
    labels = tokenized_data["labels"].to(train_device)
    resp_mask = tokenized_data["response_mask"].to(train_device)

    import numpy as np

    batch_size = 128
    sample_count = input_ids.shape[0]
    index = np.random.randint(0, sample_count, size=batch_size)

    from sft_helper import get_response_log_probs, sft_microbatch_train_step

    gradient_accumulation_steps = 8
    optimizer = torch.optim.AdamW(llm.parameters(), lr=1e-5)
    for it in range(1000):
        batch_input_ids, batch_labels, batch_resp_mask = get_batch(
            input_ids, labels, resp_mask, batch_size
        )
        results = get_response_log_probs(
            llm, batch_input_ids, batch_labels, return_token_entropy=True
        )
        log_probs = results["log_probs"]
        token_entropy = results["token_entropy"]
        loss, meta_data = sft_microbatch_train_step(
            log_probs, batch_resp_mask, gradient_accumulation_steps, 1.0
        )

        if (it + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
