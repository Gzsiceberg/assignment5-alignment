from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerBase
import torch
from datasets import load_dataset
import datasets
from tqdm import tqdm


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    max_prompt_and_output_lens = 0
    prompt_tokens_list = []
    output_tokens_list = []
    for prompt, output in tqdm(zip(prompt_strs, output_strs)):
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
    
    batch_size = len(prompt_strs)
    eos_token_id: int = tokenizer.eos_token_id # type: ignore
    input_ids = torch.full((batch_size, max_prompt_and_output_lens - 1), eos_token_id, dtype=torch.int)
    labels = input_ids.clone()
    response_mask = torch.zeros((batch_size, max_prompt_and_output_lens - 1), dtype=torch.bool)
    for idx, (prompt_tokens, output_tokens) in tqdm(enumerate(zip(prompt_tokens_list, output_tokens_list))):
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

def extract_prompt_and_response(ds: datasets.Dataset) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    responses: list[str] = []
    for data in ds:
        question: str = data["query"]  # type: ignore
        resp: str = data["response"]  # type: ignore
        prompts.append(question)
        responses.append(resp)
        if len(prompts) > 32:
            break
    return prompts, responses


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./models/Qwen/Qwen2.5-Math-1.5B")

    ds = load_dataset("hkust-nlp/dart-math-uniform")
    train: datasets.Dataset = ds["train"]  # type: ignore

    prompts, responses = extract_prompt_and_response(train)
    prompts = ['Hello, world!', 'This is a test.', 'This is another test.']
    responses = ['Hello, world!', 'This is a test.', 'This is another test.']
    tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
    from rich import print
    print(tokenized_data["input_ids"].shape)
    print(tokenized_data["response_mask"].shape)
    print(tokenized_data["labels"].shape)
    print(tokenized_data)
