from vllm import LLM, SamplingParams
from typing import Any, Callable, List
from datasets import load_dataset
import datasets
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from math_verify import parse, verify
from tqdm import tqdm
import pickle
import os
from rich import print
from dataclasses import dataclass
import torch
from cs336_alignment.extract import extract_prompt_and_response



@dataclass
class EvalEntry:
    prompt: str
    response: str
    parse_answer: str
    ground_truth: str
    reward: float
    format_reward: float
    answer_reward: float
    answer_reward_v2: float


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, Any], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    dump_data: bool = True,
) -> None:
    batch_size = 32
    # get gpu memory maximum
    gpu_memory_max = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    if gpu_memory_max >= 48:
        batch_size = 128
    elif gpu_memory_max >= 24:
        batch_size = 64
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    responses: List[str] = []
    for i in tqdm(range(num_batches), desc="generating responses"):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        outputs = vllm_model.generate(
            batch_prompts,
            sampling_params=eval_sampling_params,
        )
        for output in outputs:
            gen_text: str = output.outputs[0].text
            responses.append(gen_text)

    eval_entries: List[EvalEntry] = []
    for ground_truth, resp in tqdm(zip(ground_truths, responses), desc="evaluating responses"):
        reward_dict = reward_fn(resp, ground_truth)

        resp_answer = parse(resp)
        gt_answer = parse(ground_truth)
        is_correct = verify(resp_answer, gt_answer)

        eval_entry = EvalEntry(
            prompt=prompts[len(eval_entries)],
            response=resp,
            parse_answer=str(resp_answer),
            ground_truth=ground_truth,
            reward=reward_dict["reward"],
            format_reward=reward_dict["format_reward"],
            answer_reward=reward_dict["answer_reward"],
            answer_reward_v2=1.0 if is_correct else 0.0,
        )
        eval_entries.append(eval_entry)

    total_rewards = sum([entry.reward for entry in eval_entries])
    total_rewards_v2 = sum([entry.answer_reward_v2 for entry in eval_entries])
    total_formatting_rewards = sum([entry.format_reward for entry in eval_entries])
    total_entries = len(eval_entries)
    avg_reward = total_rewards / total_entries if total_entries > 0 else 0.0
    avg_reward_v2 = total_rewards_v2 / total_entries if total_entries > 0 else 0.0
    avg_formatting_reward = total_formatting_rewards / total_entries if total_entries > 0 else 0.0
    print(f"Average Reward over {total_entries} samples: {avg_reward}")
    print(f"Average Reward v2 over {total_entries} samples: {avg_reward_v2}")
    print(f"Average Formatting Reward over {total_entries} samples: {avg_formatting_reward}")

    if not dump_data:
        return

    with open("data/math_baseline_eval_results.pkl", "wb") as f:
        pickle.dump(eval_entries, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=512, help="Number of samples to evaluate on."
    )
    parser.add_argument(
        "--offset", type=int, default=1024, help="Offset for selecting samples."
    )
    parser.add_argument("-m", "--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Model ID to evaluate.")
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)

    ds = load_dataset("hkust-nlp/dart-math-uniform")
    train: datasets.Dataset = ds["train"] # type: ignore
    
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"gpu count: {gpu_count}")
    assert gpu_count >= 1, "At least one GPU is required."

    prompts, ground_truths = extract_prompt_and_response(train, args.limit, args.offset)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=4096, stop=["\n"]
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    model = LLM(model=f"models/{args.model_id}")

    evaluate_vllm(
        vllm_model=model,
        reward_fn=lambda resp, gt: r1_zero_reward_fn(resp, gt, False),
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


# %%
