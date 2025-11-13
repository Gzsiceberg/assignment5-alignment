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
{response}"""


def compute_logp_delta(
    model: torch.nn.Module, input_ids: torch.Tensor, resp_mask: torch.Tensor
) -> torch.Tensor:
    """Compute log prob delta between chosen and rejected responses.

    Args:
        model: The language model
        input_ids: (batch_size, seq_len) where batch_size must be even.
                   First half are chosen responses, second half are rejected.
        resp_mask: (batch_size, seq_len) mask indicating response tokens

    Returns:
        Scalar tensor with log prob delta
    """
    batch_size, seq_len = input_ids.shape
    assert batch_size % 2 == 0, "Batch size must be even."
    half_batch_size = batch_size // 2

    input_ids = input_ids.to(model.device)
    resp_mask = resp_mask.to(model.device)

    # Get log probs for all tokens (batch_size, seq_len-1)
    logp = get_response_log_probs(model, input_ids[:, :-1], input_ids[:, 1:])[
        "log_probs"
    ]

    # Mask to only include response tokens (batch_size, seq_len-1)
    masked_logp = logp * resp_mask[:, 1:]

    # Sum over sequence dimension for each example
    logp_chosen = masked_logp[:half_batch_size].sum(dim=-1)  # (half_batch_size,)
    logp_rejected = masked_logp[half_batch_size:].sum(dim=-1)  # (half_batch_size,)

    # Return delta for each pair
    logp_delta = logp_chosen - logp_rejected  # (half_batch_size,)
    return logp_delta


def tokenizer_encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts and responses, creating proper response masks.

    Args:
        tokenizer: The tokenizer
        prompts: List of prompt strings
        responses: List of response strings

    Returns:
        input_ids_batch: (batch_size, max_seq_len) padded token IDs
        response_mask: (batch_size, max_seq_len) mask where 1 = response token, 0 = prompt/padding
    """
    input_ids_list = []
    response_mask_list = []

    eos_token_id: int = tokenizer.eos_token_id  # type: ignore
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response separately
        content = prompt_template.format(instruction=prompt, response=response)
        all_tokens = tokenizer.encode(content)
        all_tokens.append(eos_token_id)

        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        # Create mask: 1 for response tokens, 0 for prompt tokens
        response_mask = torch.ones(len(all_tokens), dtype=torch.float)

        input_ids_list.append(input_ids)
        response_mask_list.append(response_mask)

    # Pad sequences
    pad_value = 0
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=float(pad_value)
    ).long()

    response_mask_batch = torch.nn.utils.rnn.pad_sequence(
        response_mask_list, batch_first=True, padding_value=0.0
    )

    return input_ids_batch, response_mask_batch


def dpo_loss_batch(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids_batch: torch.Tensor,
    response_mask_batch: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    # Compute log prob deltas (one forward pass per model)
    logp_policy_delta = compute_logp_delta(
        policy_model, input_ids_batch, response_mask_batch
    )

    with torch.inference_mode():
        logp_ref_delta = compute_logp_delta(
            ref_model, input_ids_batch, response_mask_batch
        )
        logp_ref_delta = logp_ref_delta.to(logp_policy_delta.device)

    # DPO loss: -log sigmoid(beta * (log_pi_chosen/rejected - log_ref_chosen/rejected))
    loss = -F.logsigmoid(beta * (logp_policy_delta - logp_ref_delta))
    return loss.mean()  # Average over batch (should be single value for one example)


def dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Compute DPO loss for a single prompt with chosen/rejected responses.

    More efficient implementation that does one forward pass per model.
    """
    # Batch encode: [chosen, rejected]
    input_ids_batch, response_mask_batch = tokenizer_encode_batch(
        tokenizer, [prompt, prompt], [response_chosen, response_rejected]
    )
    return dpo_loss_batch(
        policy_model,
        ref_model,
        input_ids_batch,
        response_mask_batch,
        beta,
    )


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

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
