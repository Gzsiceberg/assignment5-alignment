from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from vllm import LLM
from vllm.sampling_params import SamplingParams
import torch
from datasets import load_dataset
import datasets
from tqdm import tqdm
from jaxtyping import Float, Int, jaxtyped, Bool
from cs336_alignment.extract import extract_prompt_and_response_with_format


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    sum_tensor = torch.sum(tensor * mask, dim=dim)
    return sum_tensor / normalize_constant


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Int[torch.Tensor, "batch seq_len"],
    labels: Int[torch.Tensor, "batch seq_len"],
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids=input_ids).logits
    gathered = logits.gather(dim=-1, index=labels.long().unsqueeze(-1)).squeeze(-1)
    row_logsumexp = torch.logsumexp(logits, dim=-1)
    log_probs_for_labels = gathered - row_logsumexp

    result: dict[str, torch.Tensor] = {"log_probs": log_probs_for_labels}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss /= gradient_accumulation_steps
    loss /= batch_size
    # with torch.autocast(device_type="cuda:0", enabled=False):
    loss.backward()
    return loss, {}


def sft_microbatch_eval_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_constant: float = 1.0,
) -> torch.Tensor:

    batch_size = policy_log_probs.shape[0]
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss /= batch_size
    return loss


def _tokenize_data(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
):
    max_prompt_and_output_lens = 0
    prompt_tokens_list = []
    output_tokens_list = []
    for prompt, output in tqdm(
        zip(prompt_strs, output_strs),
        total=len(prompt_strs),
        desc="Tokenizing prompts and outputs",
    ):
        prompt_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(output)
        max_prompt_and_output_lens = max(
            max_prompt_and_output_lens, len(prompt_tokens) + len(output_tokens)
        )
        prompt_tokens_list.append(prompt_tokens)
        output_tokens_list.append(output_tokens)

    if __name__ == "__main__":
        from rich import print

        print(f"Max prompt and output length: {max_prompt_and_output_lens}")
        print(f"Sample prompt tokens: {prompt_tokens_list[0]}")
        print(f"Sample output tokens: {output_tokens_list[0]}")
        print(f"eos token id: {tokenizer.eos_token_id}")

    return prompt_tokens_list, output_tokens_list, max_prompt_and_output_lens


def tokenize_to_np(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
    data_type: str,
):
    import numpy as np

    prompt_tokens_list, output_tokens_list, max_prompt_and_output_lens = _tokenize_data(
        prompt_strs, output_strs, tokenizer
    )

    batch_size = len(prompt_strs)
    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
    input_ids = np.memmap(
        f"data/input_ids_{data_type}.npy",
        mode="w+",
        shape=(batch_size, max_prompt_and_output_lens - 1),
        dtype=np.int32,
    )
    input_ids.fill(eos_token_id)
    labels = np.memmap(
        f"data/labels_{data_type}.npy",
        mode="w+",
        shape=(batch_size, max_prompt_and_output_lens - 1),
        dtype=np.int32,
    )
    labels.fill(eos_token_id)
    response_mask = np.memmap(
        f"data/response_mask_{data_type}.npy",
        mode="w+",
        shape=(batch_size, max_prompt_and_output_lens - 1),
        dtype=bool,
    )
    response_mask.fill(False)

    for idx, (prompt_tokens, output_tokens) in tqdm(
        enumerate(zip(prompt_tokens_list, output_tokens_list)),
        total=batch_size,
        desc="Saving input_ids, labels, and response_mask",
    ):
        prompt_len = len(prompt_tokens)
        all_tokens = prompt_tokens + output_tokens
        if len(all_tokens) < max_prompt_and_output_lens:
            input_tokens = all_tokens[:]
            label_tokens = all_tokens[1:]
        else:
            input_tokens = all_tokens[:-1]
            label_tokens = all_tokens[1:]
        input_id = np.array(input_tokens, dtype=np.int32)
        label = np.array(label_tokens, dtype=np.int32)
        input_ids[idx, : len(input_id)] = input_id
        labels[idx, : len(label)] = label
        response_mask[idx, prompt_len - 1 : len(label)] = True

    input_ids.flush()
    labels.flush()
    response_mask.flush()
    del input_ids
    del labels
    del response_mask


def tokenize_to_tensor(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    prompt_tokens_list, output_tokens_list, max_prompt_and_output_lens = _tokenize_data(
        prompt_strs, output_strs, tokenizer
    )

    batch_size = len(prompt_strs)
    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
    input_ids = torch.full(
        (batch_size, max_prompt_and_output_lens - 1), eos_token_id, dtype=torch.int
    )
    labels = input_ids.clone()
    response_mask = torch.zeros(
        (batch_size, max_prompt_and_output_lens - 1), dtype=torch.bool
    )
    for idx, (prompt_tokens, output_tokens) in tqdm(
        enumerate(zip(prompt_tokens_list, output_tokens_list)),
        total=batch_size,
        desc="Creating input_ids, labels, and response_mask",
    ):
        prompt_len = len(prompt_tokens)
        all_tokens = prompt_tokens + output_tokens
        if len(all_tokens) < max_prompt_and_output_lens:
            input_tokens = all_tokens[:]
            label_tokens = all_tokens[1:]
        else:
            input_tokens = all_tokens[:-1]
            label_tokens = all_tokens[1:]
        input_id = torch.tensor(input_tokens, dtype=torch.int)
        label = torch.tensor(label_tokens, dtype=torch.int)
        input_ids[idx, : len(input_id)] = input_id
        labels[idx, : len(label)] = label
        response_mask[idx, prompt_len - 1 : len(label)] = True
    return {"input_ids": input_ids, "response_mask": response_mask, "labels": labels}


def log_generations(
    vllm: LLM,
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
):
    import random
    from rich import print

    lenght = len(prompts)
    index = random.randint(0, lenght - 1)
    prompt = prompts[index]
    gt = ground_truths[index]
    print("=== Prompt ===")
    print(prompt)

    print("=== Response ===")
    # need generate logprobs in outputs
    outputs = vllm.generate(
        [prompt],
        sampling_params=eval_sampling_params,
    )
    response = outputs[0].outputs[0].text
    print(response)

    print("=== Ground Truth ===")
    print(gt)

    print("=== Reward ===")
    from math_verify import parse, verify
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    reward_dict = r1_zero_reward_fn(response, gt, False)
    resp_answer = parse(response)
    gt_answer = parse(gt)
    reward = reward_dict["reward"]
    format_reward = reward_dict["format_reward"]
    answer_reward_v2 = verify(resp_answer, gt_answer)
    print(
        f"Reward={reward}, FormatReward={format_reward}, AnswerRewardV2={answer_reward_v2}"
    )

    print("=== End of Log ===")
    resp_length = len(response)
    gt_length = len(gt)
    print(f"Response Length: {resp_length}, Ground Truth Length: {gt_length}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument(
        "--to-np",
        action="store_true",
        help="Store tokenized data as numpy memmap files",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen2.5-Math-1.5B")

    data_math = load_dataset("hkust-nlp/dart-math-uniform")
    data_type: str = args.type
    ds: datasets.Dataset = data_math[data_type]  # type: ignore
    prompts, responses = extract_prompt_and_response_with_format(ds, args.limit, args.offset)

    if args.to_np:
        tokenize_to_np(prompts, responses, tokenizer, data_type)
        exit(0)

    tokenized_data = tokenize_to_tensor(prompts, responses, tokenizer)
    input_ids = tokenized_data["input_ids"]
    response_mask = tokenized_data["response_mask"]
    labels = tokenized_data["labels"]

    from rich import print

    print(input_ids.shape)
    print(response_mask.shape)
    print(labels.shape)

    import numpy as np
    print("Saving tokenized data to .npy files...")
    np.save("data/input_ids_tensor.npy", input_ids.numpy())
    np.save("data/response_mask_tensor.npy", response_mask.numpy())
    np.save("data/labels_tensor.npy", labels.numpy())
