from dataclasses import dataclass
import os
import regex as re
import json
from typing import Callable
from git import List
from joblib import Memory
from datasets import load_dataset, Dataset
import torch
import pickle
from tqdm import tqdm
import typer
from vllm import LLM, SamplingParams
from cs336_alignment.logger import print_and_log

memory = Memory("data/.cache", verbose=0)


@dataclass
class EvalEntry:
    prompt: str
    response: str
    ground_truth: str
    reward: float

ft_prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def extract_mmlu_answer(response: str) -> str | None:
    if "The correct answer is" not in response:
        return None
    groups = response.split("The correct answer is")
    if len(groups) < 2:
        return None
    answer_part = groups[1].strip().strip("```").strip()

    # Extract the first word or character after "The correct answer is"
    answer = answer_part.split()[0].strip().strip(".").strip(",").strip()
    if answer in {"A", "B", "C", "D"}:
        return answer
    return None


def gen_mmlu_prompt(example: dict, is_fine_tuned: bool = False) -> dict:
    system_prompt = """# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```"""

    query_format = """Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).
    
Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:"""
    prompt = query_format.format(
        subject=example["subject"],
        question=example["question"],
        options=example["choices"],
    )
    if is_fine_tuned:
        prompt = ft_prompt_template.format(instruction=prompt)
    else:
        prompt = system_prompt.format(instruction=prompt)
    return {"prompt": prompt}


def gen_gsm8k_prompt(example: dict, is_fine_tuned: bool = False) -> dict:
    prompt_template = """{question}
Answer:"""
    if is_fine_tuned:
        prompt = ft_prompt_template.format(instruction=example["question"])
    else:
        prompt = prompt_template.format(question=example["question"])
    return {"prompt": prompt}


def mmlu_reward(response: str, ground_truth: str) -> float:
    answer = extract_mmlu_answer(response)
    if answer is None:
        return -1.0
    return 1.0 if answer == ground_truth else 0.0


def extract_gsm8k_answer(response: str) -> str | None:
    # Extract the final numerical answer from the response
    matches = re.findall(r"-?\d*\.?\d+", response.replace(",", ""))
    if len(matches) == 0:
        return None
    match_str: str = matches[-1]
    return match_str.strip().strip(".").strip(",")


def gsm8k_reward(response: str, ground_truth: str) -> float:
    from cs336_alignment.extract import extract_gsm_gt
    from math_verify import parse, verify

    answer = extract_gsm8k_answer(response)
    if answer is None:
        return -1.0
    gt = extract_gsm_gt(ground_truth)
    if gt is None:
        return -1
    # is_correct = verify(answer, gt)
    is_correct = answer == gt
    return 1.0 if is_correct else 0.0


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], float] | None = mmlu_reward,
    batch_size: int = 32,
) -> List[EvalEntry]:
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
    correct_count = 0
    failed_count = 0
    for ground_truth, resp in tqdm(
        zip(ground_truths, responses), total=len(responses), desc="evaluating responses"
    ):
        reward = reward_fn(resp, ground_truth) if reward_fn is not None else 0.0
        eval_entry = EvalEntry(
            prompt=prompts[len(eval_entries)],
            response=resp,
            ground_truth=ground_truth,
            reward=reward,
        )
        eval_entries.append(eval_entry)
        if reward < 0.0:
            failed_count += 1
        elif reward > 0.0:
            correct_count += 1
    accuracy = correct_count / len(responses)
    fail_rate = failed_count / len(responses)
    if reward_fn is not None:
        print_and_log(
            f"Evaluation on {len(responses)} examples: accuracy={accuracy*100:.2f}% failrate={fail_rate*100:.2f}%"
        )
    return eval_entries


def get_evaluation_sample_params(
    n: int = 1,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    stop: list[str] | None = None,
) -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        min_tokens=4,
        seed=42,
        n=n,
    )
    sampling_params.stop = stop
    sampling_params.include_stop_str_in_output = False
    return sampling_params


def main(
    model_id: str = typer.Argument(
        "models/LLM-Research/Meta-Llama-3.1-8B", help="Model ID"
    ),
    dataset: str = typer.Option("mmlu", "-d", help="Dataset ID"),
    split: str = typer.Option(
        "test", "-s", help="Dataset split to use (train, dev, test)"
    ),
    limit: int = typer.Option(
        0, "-l", help="Limit number of evaluation samples (0 means no limit)"
    ),
    batch_size: int = typer.Option(32, "-b", help="Batch size for evaluation"),
):
    is_fine_tuned: bool = "fine-tuned" in model_id.lower()
    ground_truths: List[str] = []
    prompts: List[str] = []
    eval_sampling_params = None
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        test: Dataset = ds[split]  # type: ignore
        if limit > 0:
            test = test.select(range(limit))
        test = test.map(lambda x : gen_mmlu_prompt(x, is_fine_tuned=is_fine_tuned))
        options = ["A", "B", "C", "D"]
        ground_truths = list(map(lambda x: options[x], test["answer"]))
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=1024, temperature=0.0, stop=["```"]
        )
        prompts = test["prompt"]
    elif dataset == "gsm8k":
        ds = load_dataset("data/gsm8k")
        if split == "dev":
            split = "test"
        test: Dataset = ds[split]  # type: ignore
        if limit > 0:
            test = test.select(range(limit))
        test = test.map(lambda x : gen_gsm8k_prompt(x, is_fine_tuned=is_fine_tuned))
        ground_truths = test["answer"]
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=1024, temperature=0.0, stop=None
        )
        prompts = test["prompt"]
    elif dataset == "alpaca":
        ds = load_dataset("data/alpaca_eval")
        test = ds["test"] # type: ignore
        if limit > 0:
            test = test.select(range(limit))
        if is_fine_tuned:
            test = test.map(lambda x: {"prompt": ft_prompt_template.format(instruction=x["instruction"])})
            prompts = test["prompt"]
        else:
            prompts = test["instruction"]
        ground_truths = test["output"] 
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=1024, temperature=0.0, stop=None
        )
    elif dataset == "sst":
        ds = load_dataset("data/simple_safety_tests")
        test = ds["train"] # type: ignore
        if limit > 0:
            test = test.select(range(limit))  # type: ignore
        if is_fine_tuned:
            test = test.map(lambda x: {"prompt": ft_prompt_template.format(instruction=x["prompts_final"])})
            prompts = test["prompt"]
        else:
            prompts = test["prompts_final"]
        ground_truths = [""] * len(prompts) # dummy ground truths for sst
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=1024, temperature=0.0, stop=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    assert len(prompts) == len(ground_truths)
    assert eval_sampling_params is not None

    from vllm_util import init_vllm

    vlm = init_vllm(
        model_id=model_id,
        device="cuda:0",
        seed=42,
    )

    reward_fn = None
    if dataset == "mmlu":
        reward_fn = mmlu_reward
    elif dataset == "gsm8k":
        reward_fn = gsm8k_reward

    import time
    start_time = time.time()
    eval_entries = evaluate_vllm(
        vllm_model=vlm,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=eval_sampling_params,
        reward_fn=reward_fn,
        batch_size=batch_size,
    )
    end_time = time.time()
    tps = len(prompts) / (end_time - start_time)
    print_and_log(f"Total evaluation time: {end_time - start_time:.2f} seconds ({tps:.2f} samples/second)")

    with open(f"data/{dataset}_eval_results.pkl", "wb") as f:
        pickle.dump(eval_entries, f)
    
    model_name = os.path.basename(os.path.dirname(model_id)).lower().replace("/", "_")
    if dataset == "sst":
        print_and_log(f"Saving SST eval results for model {model_name}")
        output = []
        for example, entry in zip(test, eval_entries):
            output.append({
                "prompts": example["prompts_final"], # type: ignore
                "generator": model_name,
                "output": entry.response,
            })
        with open(f"data/sst_{model_name}.json", "w") as f:
            json.dump(output, f, indent=4)
    elif dataset == "alpaca":
        print_and_log(f"Saving Alpaca eval results for model {model_name}")
        output = []
        for example in test:
            output.append({
                "instruction": example["instruction"], # type: ignore
                "output": example["output"],        # type: ignore
                "generator": example["generator"], # type: ignore
                "dataset": example["dataset"]       # type: ignore
            })
        with open(f"data/alpaca_reference.json", "w") as f:
            json.dump(output, f, indent=4)

        output = []
        for entry, test_entry in zip(eval_entries, test):
            output.append({
                "instruction": entry.prompt,
                "output": entry.response,
                "generator": model_name,
                "dataset": test_entry["dataset"] # type: ignore
            })
        with open(f"data/alpaca_{model_name}.json", "w") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    from cs336_alignment.sft import cleanup

    try:
        typer.run(main)
    finally:
        cleanup()
