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

def compute_logp_delta(model: torch.nn.Module,
                       good_input_ids: torch.Tensor,
                       good_labels: torch.Tensor,
                       bad_input_ids: torch.Tensor,
                       bad_labels: torch.Tensor) -> torch.Tensor:
    good_input_ids = good_input_ids.to(model.device)
    good_labels = good_labels.to(model.device)

    bad_input_ids = bad_input_ids.to(model.device)
    bad_labels = bad_labels.to(model.device)
    
    logp_good = get_response_log_probs(model, good_input_ids, good_labels)["log_probs"]
    logp_bad = get_response_log_probs(model, bad_input_ids, bad_labels)["log_probs"]
    logp_delta = (logp_good - logp_bad).sum(-1)
    return logp_delta
    

def _tokenizer_encode(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    full_prompt = prompt_template.format(instruction=prompt, response=response)
    input_ids = tokenizer.encode(full_prompt)
    labels = input_ids[1:] + [tokenizer.eos_token_id]
    assert len(input_ids) == len(labels)
    return torch.tensor(input_ids), torch.tensor(labels)

def dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    good_input_ids, good_labels = _tokenizer_encode(tokenizer, prompt, response_chosen)
    bad_input_ids, bad_labels = _tokenizer_encode(tokenizer, prompt, response_rejected)

    logp_policy_delta = compute_logp_delta(policy_model, good_input_ids, good_labels, bad_input_ids, bad_labels)
    with torch.inference_mode():
        logp_ref_delta = compute_logp_delta(ref_model, good_input_ids, good_labels, bad_input_ids, bad_labels)
        logp_ref_delta = logp_policy_delta.to(logp_ref_delta.device)
    
    loss = -F.logsigmoid(beta * (logp_policy_delta - logp_ref_delta))
    return loss

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained(f"{FIXTURES_PATH}/tiny-gpt2")
    model_ref = AutoModelForCausalLM.from_pretrained(f"{FIXTURES_PATH}/tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = dpo_loss(
        model,
        model_ref,
        tokenizer,
        0.5,
        prompt,
        good_response,
        bad_response,
    )
    print(f"DPO loss: {loss.item():.4f}")
