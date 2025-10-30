from vllm import LLM, SamplingParams
from typing import Callable, List
from datasets import load_dataset
import datasets
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tqdm import tqdm
import regex as re
import pickle
import os
from rich import print


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    batch_size = 32
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    responses = []
    for i in tqdm(range(num_batches)):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        outputs = vllm_model.generate(
            batch_prompts,
            sampling_params=eval_sampling_params,
        )
        for output in outputs:
            gen_text: str = output.outputs[0].text
            responses.append(gen_text)

    all_rewards = []
    total_rewards = 0.0
    for ground_truth, resp in zip(ground_truths, responses):
        reward_dict = reward_fn(resp, ground_truth)
        total_rewards += reward_dict.get("reward", 0.0)
        all_rewards.append(reward_dict)

    avg_reward = total_rewards / len(prompts) if prompts else 0.0
    print(f"Average Reward over {len(prompts)} samples: {avg_reward}")

    eval_data = {
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
        "rewards": all_rewards,
    }
    with open("data/math_baseline_eval_results.pkl", "wb") as f:
        pickle.dump(eval_data, f)


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


def generate_prompt_and_gt(ds: datasets.Dataset) -> tuple[List[str], List[str]]:
    prompt_templ = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {0}
Assistant: <think>"""
    prompts = []
    ground_truths = []
    for t, data in tqdm(enumerate(ds)):
        question = data["question"]
        answer_text = data["answer"]
        answer = extract_answer(answer_text)
        assert answer is not None, f"Could not extract answer from: {answer_text}"
        full_prompt = prompt_templ.format(question)
        prompts.append(full_prompt)
        ground_truths.append(answer)
    return prompts, ground_truths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=-1, help="Number of samples to evaluate on."
    )
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main")
    train: datasets.Dataset = ds["train"]
    if args.limit > 0:
        train = train.select(range(args.limit))

    prompts, ground_truths = generate_prompt_and_gt(train)
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    model = LLM(model="models/Qwen/Qwen2.5-Math-1.5B")

    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
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
