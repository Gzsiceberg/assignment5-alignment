
from dataclasses import dataclass
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

def gen_prompt(example: dict) -> dict:
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
        options=example["choices"]
    )
    return {"prompt": system_prompt.format(instruction=prompt)}


def evaluate_vllm(
    vllm_model: LLM,
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
    correct_count = 0
    for ground_truth, resp in tqdm(zip(ground_truths, responses), total=len(responses), desc="evaluating responses"):
        answer = extract_answer(resp)
        eval_entry = EvalEntry(
            prompt=prompts[len(eval_entries)],
            response=resp,
            parse_answer=answer,
            ground_truth=ground_truth,
        )
        eval_entries.append(eval_entry)
        if answer is not None and answer == ground_truth:
            correct_count += 1
    accuracy = correct_count / len(responses)
    print_and_log(f"Evaluation accuracy: {accuracy*100:.2f}%")

    if not dump_data:
        return

    with open("data/mmlu_eval_results.pkl", "wb") as f:
        pickle.dump(eval_entries, f)
    


def get_evaluation_sample_params(n: int = 1, max_tokens: int = 2048) -> SamplingParams:
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=max_tokens, min_tokens=4, stop=["\n"], seed=42,
        n=n,
    )
    sampling_params.stop = ["# Query:"]
    sampling_params.include_stop_str_in_output = False
    return sampling_params


def main(model_id: str = typer.Argument("models/LLM-Research/Meta-Llama-3.1-8B", help="Model ID"),
         split: str = typer.Option("dev", "-s", help="Dataset split to use (train, dev, test)")):
    ds = load_dataset("cais/mmlu", "all")
    test: Dataset = ds[split] # type: ignore
    test = test.map(gen_prompt)

    from vllm_util import init_vllm
    prompts = test["prompt"]
    ground_truths = test["answer"]
    eval_sampling_params = get_evaluation_sample_params(1, max_tokens=1024)
    print_and_log(f"Loaded {len(prompts)} evaluation samples from MMLU {split} split.")

    vlm = init_vllm(
        model_id=model_id,
        device="cuda:0",
        seed=42,
    )
    evaluate_vllm(
        vllm_model=vlm,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=eval_sampling_params,
        dump_data=True,
    )


if __name__ == "__main__":
    from cs336_alignment.sft import cleanup
    try:
        typer.run(main)
    finally:
        cleanup()





