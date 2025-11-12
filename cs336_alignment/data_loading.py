import os
import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from cs336_alignment.logger import setup_logging, print_and_log
import numpy as np

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, dataset_path: str | os.PathLike, seq_len: int, shuffle=True):
        ds = load_dataset("json", data_files=str(dataset_path))
        d: Dataset = ds["train"]  # type: ignore
        d = d.map(
            lambda x: {
                "input": prompt_template.format(
                    instruction=x["prompt"], response=x["response"]
                )
            },
            desc="Creating input prompts",
        )
        if shuffle:
            d = d.shuffle(seed=42)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle

        eos_token_id: int = tokenizer.eos_token_id  # type: ignore
        print_and_log(f"Using eos_token_id: {eos_token_id}")
        d = d.map(lambda x: {"tokens": tokenizer.encode(x["input"]) + [eos_token_id]}, desc="Tokenizing")
        total_tokens = 0
        for item in d["tokens"]:
            total_tokens += len(item)
        print_and_log(f"Total tokens in dataset: {total_tokens}")
        
        all_tokens = np.zeros((total_tokens,), dtype=np.int32)
        start_idx = 0
        for tokens in d["tokens"]:
            end_idx = start_idx + len(tokens)
            all_tokens[start_idx:end_idx] = np.array(tokens, dtype=np.int32)
            start_idx = end_idx
        
        input_ids_list = []
        labels_list = []
        num_segments = total_tokens // seq_len
        for i in range(num_segments):
            input_ids = all_tokens[i * seq_len : (i + 1) * seq_len]
            labels = all_tokens[i * seq_len + 1 : (i + 1) * seq_len + 1]
            assert len(input_ids) == len(labels)
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        self.input_ids_list = input_ids_list
        self.labels_list = labels_list

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids_list[idx], "labels": self.labels_list[idx]}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "models/LLM-Research/Meta-Llama-3.1-8B/"
    )
    dataset = SFTDataset(
        tokenizer,
        dataset_path="data/sft/train.jsonl",
        seq_len=512,
        shuffle=True,
    )
