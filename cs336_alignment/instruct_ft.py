import shutil
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import os
import torch
from datasets import load_dataset, Dataset
from cs336_alignment.config import InstructionFineTuningConfig, load_config_from_file
from cs336_alignment.logger import setup_logging, print_and_log
import random
import numpy as np
import typer
from tqdm import tqdm, trange
from cs336_alignment.data_loading import SFTDataset
import torch.nn.functional as F


def evaluate_model_on_dataset(
    llm: PreTrainedModel,
    loader: torch.utils.data.DataLoader,
    device: str,
    prob_filter: float = 0.0,
):
    llm.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_count: int = 0
    with torch.inference_mode(True):
        for batch in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
            if prob_filter > 0.0 and total_count > 0:
                if random.random() > prob_filter:
                    continue
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)
            logits: torch.Tensor = llm(input_ids).logits  # type: ignore

            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            assert loss.dim() == 0, f"Loss dimension expected to be 0, got {loss.dim()}"
            total_loss += loss
            total_count += 1

    avg_loss = total_loss.item() / total_count
    perplexity = np.exp(avg_loss)
    print_and_log(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    llm.train()


def save_checkpoint(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_iter: int,
    checkpoint_path: str,
):
    print_and_log(f"Saving checkpoint to {checkpoint_path} at iteration {last_iter}")
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "last_iter": last_iter,
    }
    torch.save(checkpoint, checkpoint_path)
    print_and_log(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str,
) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    print_and_log(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint["last_iter"]


def main(
    config_path: str = typer.Argument(
        "config/instruction/t.yaml", help="Path to configuration file"
    ),
    seed: int = typer.Option(42, "-s", help="Random seed for reproducibility"),
    limit: int = typer.Option(
        0, "-l", help="Limit the number of training samples (0 for no limit)"
    ),
    resume: bool = typer.Option(False, "-r", help="Resume training from checkpoint"),
    shutdown: bool = typer.Option(False, "-d", help="Shutdown after training"),
):

    config = load_config_from_file(config_path)
    config = InstructionFineTuningConfig(**config)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    setup_logging(f"instruct_ft_{config_name}.log")
    print_and_log(f"{config}")
    if shutdown:
        print_and_log("Shutdown after training is enabled.")
    batch_size = config.batch_size
    max_epochs = config.epochs
    gradient_accumulation_steps = config.gradient_accumulation_steps

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    is_test: bool = limit > 0

    model_id = config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_device = "cuda:0"
    checkpoint_dir = f"{model_id}-fine-tuned"
    llm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id if not resume else checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": train_device},
    )

    seq_len = config.seq_len
    dataset = SFTDataset(
        tokenizer,
        dataset_path="data/sft/train.jsonl",
        seq_len=seq_len,
        shuffle=True,
        limit=limit,
    )

    val_dataset = SFTDataset(
        tokenizer,
        dataset_path="data/sft/test.jsonl",
        seq_len=seq_len,
        shuffle=False,
        limit=limit,
    )

    total_samples = len(dataset)
    print_and_log(
        f"Total training samples: {total_samples} Total validation samples: {len(val_dataset)}"
    )
    loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, drop_last=True
    )

    if config.use_compile:
        torch.set_float32_matmul_precision("high")
        llm = torch.compile(llm)  # type: ignore

    assert (
        batch_size % gradient_accumulation_steps == 0
    ), "Batch size must be divisible by gradient accumulation steps"
    micro_batch_size = batch_size // gradient_accumulation_steps
    train_steps = len(loader) * max_epochs
    vocab_size = tokenizer.vocab_size
    print_and_log(
        f"Vocabulary size: {vocab_size:,}, Training steps: {train_steps:,}, Micro-batch size: {micro_batch_size}"
    )

    optimizer = torch.optim.AdamW(llm.parameters(), lr=config.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)  # type: ignore
    from transformers import get_cosine_schedule_with_warmup  # type: ignore

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_steps * 0.03),
        num_training_steps=train_steps,
    )

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        llm.save_pretrained(checkpoint_dir)  # type: ignore

    checkpoint_path = f"{checkpoint_dir}/checkpoint.pth"
    if resume:
        last_train_iter = load_checkpoint(optimizer, lr_scheduler, checkpoint_path)
        lr = lr_scheduler.get_last_lr()[0]
        print_and_log(
            f"Resuming training from checkpoint. lr={lr:.6f}, last_iter={last_train_iter}"
        )
    else:
        last_train_iter = 0

    if is_test:
        print_and_log("Running in test mode with limited data. Skipping training.")
        evaluate_model_on_dataset(llm, val_loader, train_device)

    eval_interval = train_steps // config.eval_number
    if eval_interval == 0:
        eval_interval = train_steps // 10
    print_and_log(
        f"Evaluation interval set to every {eval_interval} training iterations."
    )

    start_time = time.time()
    moving_avg_loss = torch.tensor(0.0, device=train_device)
    for epoch in tqdm(range(max_epochs)):
        iter_per_epoch = len(loader)
        for itr, batch in (
            tpar := tqdm(
                enumerate(loader), total=iter_per_epoch, desc=f"Epoch {epoch+1}"
            )
        ):
            if itr < last_train_iter:
                continue
            input_ids: torch.Tensor = batch["input_ids"]
            labels: torch.Tensor = batch["labels"]
            assert input_ids.shape == (
                batch_size,
                seq_len,
            ), f"input_ids shape: {input_ids.shape}"
            assert labels.shape == (
                batch_size,
                seq_len,
            ), f"labels shape: {labels.shape}"

            total_loss = do_grad_accumulate(
                gradient_accumulation_steps,
                train_device,
                llm,
                seq_len,
                micro_batch_size,
                input_ids,
                labels,
            )

            # Update moving average of loss
            with torch.no_grad():
                if itr == 0 and epoch == 0:
                    moving_avg_loss = total_loss.detach()
                else:
                    moving_avg_loss = (
                        0.9 * moving_avg_loss.detach() + 0.1 * total_loss.detach()
                    )
            grad_norm = torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0)

            current_lr = lr_scheduler.get_last_lr()[0]
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if itr % 10 == 0:
                tpar.set_description(f"Loss: {moving_avg_loss.item():.4f}")
                print_and_log(
                    f"Epoch {epoch+1} Iter {itr+1}/{iter_per_epoch}, Loss: {moving_avg_loss.item():.4f}, Grad Norm: {grad_norm.item():.4f}, LR: {current_lr:.6f}"
                )

            last_train_iter += 1
            if (itr + 1) % eval_interval == 0 or (itr == 10 and not resume):
                print_and_log(
                    f"--- Evaluation at Epoch {epoch+1} Iteration {itr+1} ---"
                )
                evaluate_model_on_dataset(
                    llm, val_loader, train_device, 0.05 if not is_test else 0.0
                )
                # Save checkpoint after evaluation
                safe_save_checkpoint(
                    checkpoint_dir, llm, optimizer, lr_scheduler, last_train_iter
                )

    print_and_log("Final evaluation after training completion:")
    evaluate_model_on_dataset(llm, val_loader, train_device)

    if not is_test:
        # Save the fine-tuned model
        save_checkpoint(
            optimizer,
            lr_scheduler,
            last_iter=last_train_iter,
            checkpoint_path=checkpoint_path,
        )
        llm.save_pretrained(checkpoint_dir)  # type: ignore
        tokenizer.save_pretrained(checkpoint_dir)
        print_and_log(f"Fine-tuned model saved to {checkpoint_dir}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_and_log(f"Training completed in {elapsed_time/60:.2f} minutes.")

    if shutdown:
        print_and_log("Shutting down the system as requested.")
        os.system("runpodctl stop pod $RUNPOD_POD_ID")


def safe_save_checkpoint(
    checkpoint_dir: str,
    llm: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    last_train_iter: int,
):
    save_start_time = time.time()
    tmp_checkpoint_dir = f"{checkpoint_dir}_temp"
    if not os.path.exists(tmp_checkpoint_dir):
        os.makedirs(tmp_checkpoint_dir, exist_ok=True)
    tmp_checkpoint_path = f"{tmp_checkpoint_dir}/checkpoint.pth"

    llm.save_pretrained(tmp_checkpoint_dir)  # type: ignore
    save_checkpoint(
        optimizer,
        lr_scheduler,
        last_iter=last_train_iter,
        checkpoint_path=tmp_checkpoint_path,
    )
    # rename temp directory to main checkpoint directory
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.rename(tmp_checkpoint_dir, checkpoint_dir)
    save_end_time = time.time()
    print_and_log(
        f"Model checkpoint saved to {checkpoint_dir} after evaluation. Time taken: {save_end_time - save_start_time:.2f} seconds."
    )


def do_grad_accumulate(
    gradient_accumulation_steps: int,
    train_device: str,
    llm: PreTrainedModel,
    seq_len: int,
    micro_batch_size: int,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=train_device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for s in trange(gradient_accumulation_steps, desc="Micro-batches", leave=False):
            start_idx = s * micro_batch_size
            end_idx = (s + 1) * micro_batch_size
            input_ids_mb = input_ids[start_idx:end_idx].to(train_device)
            labels_mb = labels[start_idx:end_idx].to(train_device)
            logits: torch.Tensor = llm(input_ids_mb).logits  # type: ignore
            vocab_size = logits.size(-1)
            assert logits.shape == (
                micro_batch_size,
                seq_len,
                vocab_size,
            ), f"logits shape: {logits.shape}"
            loss = (
                F.cross_entropy(logits.view(-1, vocab_size), labels_mb.view(-1))
                / gradient_accumulation_steps
            )
            loss.backward()
            total_loss += loss.detach()
    return total_loss


if __name__ == "__main__":
    typer.run(main)
