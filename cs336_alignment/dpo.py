import os
import torch
from transformers import PreTrainedTokenizerBase
from cs336_alignment.logger import setup_logging, print_and_log
import random
import numpy as np
import typer
from tqdm import tqdm, trange
import torch.nn.functional as F
from cs336_alignment.sft_helper import get_response_log_probs


prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}
"""


def dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    good_prompt = prompt_template.format(instruction=prompt, response=response_chosen)
    bad_prompt = prompt_template.format(instruction=prompt, response=response_rejected)
    good_input_ids = tokenizer.encode(good_prompt)
    good_labels = good_input_ids[1:] + [tokenizer.eos_token_id]
    assert len(good_input_ids) == len(good_labels)
    bad_input_ids = tokenizer.encode(bad_prompt)
    bad_labels = bad_input_ids[1:] + [tokenizer.eos_token_id]
    assert len(bad_input_ids) == len(bad_labels)

    good_input_ids = torch.tensor([good_input_ids]).to(policy_model.device)
    good_labels = torch.tensor([good_labels]).to(policy_model.device)
    logp_policy_good = get_response_log_probs(
        policy_model, good_input_ids, good_labels
    )["log_probs"]

    bad_input_ids = torch.tensor([bad_input_ids]).to(policy_model.device)
    bad_labels = torch.tensor([bad_labels]).to(policy_model.device)
    logp_policy_bad = get_response_log_probs(policy_model, bad_input_ids, bad_labels)[
        "log_probs"
    ]

    logp_policy_delta = logp_policy_good - logp_policy_bad

    with torch.inference_mode():
        good_input_ids = good_input_ids.to(ref_model.device)
        good_labels = good_labels.to(ref_model.device)
        logp_ref_good = get_response_log_probs( ref_model, good_input_ids, good_labels)["log_probs"]

        bad_input_ids = bad_input_ids.to(ref_model.device)
        bad_labels = bad_labels.to(ref_model.device)

        logp_ref_bad = get_response_log_probs(ref_model, bad_input_ids, bad_labels)["log_probs"]
        logp_ref_delta = logp_ref_good - logp_ref_bad
    
    loss = -F.logsigmoid(beta * (logp_policy_delta - logp_ref_delta))
    return loss
