from dataclasses import dataclass
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
    parse_answer: str | None
    ground_truth: str
    reward: float


def extract_answer(response: str) -> str | None:
    if "The correct answer is" not in response:
        return None
    groups = response.split("The correct answer is")
    if len(groups) < 2:
        return None
    answer_part = groups[1].strip()

    # Extract the first word or character after "The correct answer is"
    answer = answer_part.split()[0].strip().strip(".").strip(",")
    if answer in {"A", "B", "C", "D"}:
        return answer
    return None


def gen_mmlu_prompt(example: dict) -> dict:
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
    return {"prompt": system_prompt.format(instruction=prompt)}


def gen_gsm8k_prompt(example: dict) -> dict:
    prompt = """{question}
Answer:"""
    return {"prompt": prompt.format(question=example["question"])}


def mmlu_reward(response: str, ground_truth: str) -> tuple[str | None, float]:
    answer = extract_answer(response)
    if answer is None:
        return None, 0.0
    return answer, 1.0 if answer == ground_truth else 0.0


def gsm8k_reward(response: str, ground_truth: str) -> tuple[str | None, float]:
    from cs336_alignment.extract import extract_gsm_answer

    answer = extract_gsm_answer(response)
    gt = extract_gsm_answer(ground_truth)
    if answer is None:
        return None, 0.0
    return answer, 1.0 if answer == gt else 0.0


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], tuple[str | None, float]] = mmlu_reward,
) -> List[EvalEntry]:
    batch_size = 32
    # get gpu memory maximum
    gpu_memory_max = torch.cuda.get_device_properties(0).total_memory / (
        1024**3
    )  # in GB
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
    correct_count = 0
    failed_count = 0
    for ground_truth, resp in tqdm(
        zip(ground_truths, responses), total=len(responses), desc="evaluating responses"
    ):
        answer, reward = reward_fn(resp, ground_truth)
        eval_entry = EvalEntry(
            prompt=prompts[len(eval_entries)],
            response=resp,
            parse_answer=answer,
            ground_truth=ground_truth,
            reward=reward,
        )
        eval_entries.append(eval_entry)
        if answer is None:
            failed_count += 1
        elif reward > 0.0:
            correct_count += 1
    accuracy = correct_count / len(responses)
    fail_rate = failed_count / len(responses)
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
    dataset: str = typer.Argument("mmlu", help="Dataset ID"),
    split: str = typer.Option(
        "dev", "-s", help="Dataset split to use (train, dev, test)"
    ),
    limit: int = typer.Option(
        0, "-l", help="Limit number of evaluation samples (0 means no limit)"
    ),
):
    ground_truths: List[str] = []
    eval_sampling_params = None
    if dataset == "mmlu":
        ds = load_dataset("cais/mmlu", "all")
        test: Dataset = ds[split]  # type: ignore
        if limit > 0:
            test = test.select(range(limit))
        test = test.map(gen_mmlu_prompt)
        ground_truths = test["answer"]
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=4096, temperature=0.0, stop=["# Query:"]
        )
    elif dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        if split == "dev":
            split = "test"
        test: Dataset = ds[split]  # type: ignore
        if limit > 0:
            test = test.select(range(limit))
        test = test.map(gen_gsm8k_prompt)
        ground_truths = test["answer"]
        eval_sampling_params = get_evaluation_sample_params(
            1, max_tokens=4096, temperature=0.0, stop=None
        )

    prompts = test["prompt"]  # type: ignore
    assert len(prompts) == len(ground_truths)
    assert eval_sampling_params is not None
    print_and_log(f"Loaded {len(prompts)} evaluation samples from MMLU {split} split.")

    from vllm_util import init_vllm

    vlm = init_vllm(
        model_id=model_id,
        device="cuda:0",
        seed=42,
    )

    eval_entries = evaluate_vllm(
        vllm_model=vlm,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=eval_sampling_params,
        reward_fn=mmlu_reward if dataset == "mmlu" else gsm8k_reward,
    )

    with open(f"data/{dataset}_eval_results.pkl", "wb") as f:
        pickle.dump(eval_entries, f)


if __name__ == "__main__":
    from cs336_alignment.sft import cleanup

    try:
        typer.run(main)
    finally:
        cleanup()
