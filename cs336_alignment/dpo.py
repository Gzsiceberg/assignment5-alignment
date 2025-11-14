import os
import time
import shutil
import torch
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.config import DPOConfig, load_config_from_file
from cs336_alignment.logger import setup_logging, print_and_log
import random
import numpy as np
import typer
from tqdm import tqdm, trange
import torch.nn.functional as F
from cs336_alignment.sft_helper import get_response_log_probs
from tqdm import tqdm, trange
import datasets


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


def save_checkpoint(
    optimizer: torch.optim.Optimizer,
    last_iter: int,
    checkpoint_path: str,
):
    print_and_log(f"Saving checkpoint to {checkpoint_path} at iteration {last_iter}")
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "last_iter": last_iter,
    }
    torch.save(checkpoint, checkpoint_path)
    print_and_log(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print_and_log(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint["last_iter"]


def eval_dpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    dataset: datasets.Dataset,
) -> float:
    with (
        torch.inference_mode(True),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        total_loss = torch.tensor(0.0, device=policy_model.device)
        for example in tqdm(
            dataset, total=len(dataset), desc="Evaluating", leave=False
        ):
            prompt = example["prompt"]  # type: ignore
            response_chosen = example["good"]  # type: ignore
            response_rejected = example["bad"]  # type: ignore

            loss = dpo_loss(
                policy_model,
                ref_model,
                tokenizer,
                beta,
                prompt,
                response_chosen,
                response_rejected,
            )
            total_loss += loss
    avg_loss = total_loss.item() / len(dataset)
    return avg_loss


def train(
    config_path: str = typer.Argument(
        "config/dpo_test.yaml", help="Path to config file"
    ),
    limit: int = typer.Option(
        -1, "-l", help="Limit number of training examples (-1 for no limit)"
    ),
    resume: bool = typer.Option(False, "-r", help="Resume training from checkpoint"),
    shutdown: bool = typer.Option(False, "-d", help="Shutdown after training")
):

    seed = 52
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = load_config_from_file(config_path)
    dpo_config = DPOConfig(**config)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    setup_logging(f"dpo_{config_name}.log")
    print_and_log(f"{dpo_config}")
    if shutdown:
        print_and_log("Shutdown after training is enabled.")

    is_test = "_test" in config_name.lower()

    gpu_count = torch.cuda.device_count()
    model_id = dpo_config.model_id
    use_compile = dpo_config.use_compile

    checkpoint_dir = f"{dpo_config.model_id}-dpo-finetuned"
    train_device = "cuda:0"
    ref_device = "cuda:1" if gpu_count > 1 else "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    llm: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_id if not resume else checkpoint_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": train_device},
    )

    llm_ref: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": ref_device},
    )

    if use_compile:
        torch.set_float32_matmul_precision("high")
        llm = torch.compile(llm)  # type: ignore
        llm_ref = torch.compile(llm_ref)  # type: ignore

    from cs336_alignment.dpo_data import get_full_dataset

    train_ds, test_ds = get_full_dataset()
    if limit > 0:
        train_ds = train_ds.select(range(limit))

    train_ds = train_ds.shuffle()
    test_ds = test_ds.shuffle()
    test_ds = test_ds.select(range(256))

    optimizer = torch.optim.RMSprop(llm.parameters(), lr=dpo_config.lr)  # type: ignore
    gradient_accumulation_steps = dpo_config.gradient_accumulation_steps
    training_steps = len(train_ds)
    eval_interval = training_steps // 50 if not is_test else 64
    save_interval = min(gradient_accumulation_steps * 100, training_steps)
    print_and_log(f"Training steps: {training_steps}, Eval interval: {eval_interval} save interval: {save_interval}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    start_time = time.time()
    moving_avg_loss = torch.tensor(0.0, device=train_device)
    if resume:
        last_iter: int = load_checkpoint(
            optimizer,
            checkpoint_path=f"{checkpoint_dir}/checkpoint.pth",
        )
        print_and_log(f"Resuming training from iteration {last_iter}")
    else:
        last_iter = 0
    total_loss = torch.tensor(0.0, device=train_device)
    is_first_step = True
    for itr, example in tqdm(
        enumerate(train_ds), total=training_steps, desc="Training"
    ):
        if itr < last_iter:
            continue

        prompt = example["prompt"] # type: ignore
        response_chosen = example["good"]  # type: ignore
        response_rejected = example["bad"]  # type: ignore

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = (
                dpo_loss(
                    llm,
                    llm_ref,
                    tokenizer,
                    beta=0.1,
                    prompt=prompt,
                    response_chosen=response_chosen,
                    response_rejected=response_rejected,
                )
                / gradient_accumulation_steps
            )
            total_loss += loss.detach()
            loss.backward()
        
        is_last_step = (itr + 1) == training_steps
            
        if (itr + 1) % gradient_accumulation_steps == 0 or is_last_step:
            with torch.no_grad():
                if is_first_step:
                    moving_avg_loss = total_loss
                    is_first_step = False
                else:
                    moving_avg_loss = 0.9 * moving_avg_loss + 0.1 * total_loss
            total_loss = torch.tensor(0.0, device=train_device)
            grad_norm = torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            print_and_log(
                f"Iter={itr + 1}/{training_steps} loss={moving_avg_loss.item():.4f} grad_norm={grad_norm:.4f}"
            )

        if (itr + 1) % eval_interval == 0 or itr + 1 == gradient_accumulation_steps or is_last_step:
            # Evaluation code can be added here
            eval_loss = eval_dpo_loss(
                llm,
                llm_ref,
                tokenizer,
                beta=0.1,
                dataset=test_ds,
            )
            print_and_log(f"Iter={itr + 1}/{training_steps} eval_loss={eval_loss:.4f}")
        
        if (itr + 1) % save_interval == 0 or itr + 1 == gradient_accumulation_steps:
            tokenizer.save_pretrained(checkpoint_dir)
            save_checkpoint_safe(checkpoint_dir, llm, optimizer, itr + 1)

    # Save final model
    llm.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    final_eval_loss = eval_dpo_loss(
        llm,
        llm_ref,
        tokenizer,
        beta=0.1,
        dataset=test_ds,
    )
    print_and_log(f"Final eval loss: {final_eval_loss:.4f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_hours = elapsed_time / 3600
    print_and_log(f"Total training time: {elapsed_time_hours:.2f} hours")

    if shutdown:
        print_and_log("Shutting down the system as requested.")
        os.system("runpodctl stop pod $RUNPOD_POD_ID")

def save_checkpoint_safe(checkpoint_dir, llm, optimizer, itr):
    save_start_time = time.time()
    tmp_checkpoint_dir = f"{checkpoint_dir}_temp"
    if not os.path.exists(tmp_checkpoint_dir):
        os.makedirs(tmp_checkpoint_dir, exist_ok=True)
    tmp_checkpoint_path = f"{tmp_checkpoint_dir}/checkpoint.pth"

    llm.save_pretrained(tmp_checkpoint_dir)  # type: ignore
    save_checkpoint(
                optimizer,
                last_iter=itr,
                checkpoint_path=tmp_checkpoint_path,
            )
            # rename temp directory to main checkpoint directory
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.rename(tmp_checkpoint_dir, checkpoint_dir)
    save_end_time = time.time()
    print_and_log(f"Model checkpoint saved to {checkpoint_dir} after evaluation. Time taken: {save_end_time - save_start_time:.2f} seconds")


if __name__ == "__main__":
    typer.run(train)
