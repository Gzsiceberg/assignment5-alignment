import datasets
import regex as re
from datasets import load_dataset, concatenate_datasets, Dataset


def filter_multi_conv(example):
    # filter multi-turn conversations
    chosen = example["chosen"]
    # we want exactly one "Human: " and one "Assistant: ", all need at the start of line
    human_count = len(re.findall(r"^Human: ", chosen, flags=re.MULTILINE))
    assistant_count = len(re.findall(r"^Assistant: ", chosen, flags=re.MULTILINE))
    return human_count == 1 and assistant_count == 1


def extract_prompt_and_response(example):
    chosen = example["chosen"]
    prompt_match = re.search(
        r"^Human: (.+?)^Assistant: ", chosen, flags=re.MULTILINE | re.DOTALL
    )
    response_match = re.search(
        r"^Assistant: (.+)$", chosen, flags=re.MULTILINE | re.DOTALL
    )
    prompt = prompt_match.group(1).strip() if prompt_match else ""
    resp_good = response_match.group(1).strip() if response_match else ""
    reject = example["rejected"]
    return_prompt_match = re.search(
        r"^Human: (.+?)^Assistant: ", reject, flags=re.MULTILINE | re.DOTALL
    )
    if return_prompt_match:
        return_prompt = return_prompt_match.group(1).strip()
        assert return_prompt == prompt, "Prompt in chosen and rejected do not match!"
    response_match = re.search(
        r"^Assistant: (.+)$", reject, flags=re.MULTILINE | re.DOTALL
    )
    resp_bad = response_match.group(1).strip() if response_match else ""
    return {"prompt": prompt, "good": resp_good, "bad": resp_bad}


def get_full_dataset() -> tuple[Dataset, Dataset]:
    subsets = [
        "helpful-base",
        "harmless-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ]
    all_datasets = []
    for d in subsets:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=d)
        for split in ds:
            ds[split] = ds[split].add_column("source_dir", [d] * len(ds[split]))  # type: ignore
        all_datasets.append(ds)
    train_full = concatenate_datasets([ds["train"] for ds in all_datasets])
    test_full = concatenate_datasets([ds["test"] for ds in all_datasets])
    full = datasets.DatasetDict({"train": train_full, "test": test_full})

    train = full["train"]
    test = full["test"]

    train = train.filter(filter_multi_conv)
    test = test.filter(filter_multi_conv)

    train = train.map(extract_prompt_and_response)
    test = test.map(extract_prompt_and_response)

    return train, test
