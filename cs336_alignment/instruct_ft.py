from transformers import AutoModelForCausalLM, AutoTokenizer
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

def evaluate_model_on_dataset(llm: AutoModelForCausalLM, loader: torch.utils.data.DataLoader, device: str):
    llm.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_count: int = 0
    with torch.inference_mode(True):
        for batch in tqdm(loader, total=len(loader), desc="Evaluating", leave=False):
            input_ids: torch.Tensor = batch["input_ids"].to(device)
            labels: torch.Tensor = batch["labels"].to(device)
            logits = llm(input_ids).logits  # type: ignore

            vocab_size = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            assert loss.dim() == 0, f"Loss dimension expected to be 0, got {loss.dim()}"
            total_loss += loss
            total_count += 1

    avg_loss = total_loss.item() / total_count
    perplexity = np.exp(avg_loss)
    print_and_log(f"Evaluation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    llm.train()

def main(config_path: str = typer.Argument("config/instruction/t.yaml", help="Path to configuration file"),
         seed: int = typer.Option(42, "-s", help="Random seed for reproducibility"),
         limit: int = typer.Option(0, "-l", help="Limit the number of training samples (0 for no limit)")):
    
    config = load_config_from_file(config_path)
    config = InstructionFineTuningConfig(**config)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    setup_logging(f"instruct_ft_{config_name}.log")
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
    llm: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
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
    print_and_log(f"Total training samples: {total_samples} Total validation samples: {len(val_dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)


    if config.use_compile:
        torch.set_float32_matmul_precision("high")
        llm = torch.compile(llm) # type: ignore

    assert batch_size % gradient_accumulation_steps == 0, "Batch size must be divisible by gradient accumulation steps"
    micro_batch_size = batch_size // gradient_accumulation_steps
    train_steps = len(loader) * max_epochs
    vocab_size = tokenizer.vocab_size
    print_and_log(f"Vocabulary size: {vocab_size:,}, Training steps: {train_steps:,}, Micro-batch size: {micro_batch_size}")

    optimizer = torch.optim.AdamW(llm.parameters(), lr=config.lr) # type: ignore
    from transformers import get_cosine_schedule_with_warmup  # type: ignore
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_steps * 0.03),
        num_training_steps=train_steps,
    )

    if not is_test:
        output_dir = f"{model_id}-fine-tuned"
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
    
    if is_test:
        print_and_log("Running in test mode with limited data. Skipping training.")
        evaluate_model_on_dataset(llm, val_loader, train_device)

    eval_interval = train_steps // config.eval_number
    if eval_interval == 0:
        eval_interval = train_steps // 10
    print_and_log(f"Evaluation interval set to every {eval_interval} training iterations.")
    import time
    start_time = time.time()
    for epoch in tqdm(range(max_epochs)):
        for batch_idx, batch in (tpar:= tqdm(enumerate(loader), total=len(loader))):
            input_ids: torch.Tensor = batch["input_ids"].to(train_device)
            labels: torch.Tensor = batch["labels"].to(train_device)
            assert input_ids.shape == (batch_size, seq_len), f"input_ids shape: {input_ids.shape}"
            assert labels.shape == (batch_size, seq_len), f"labels shape: {labels.shape}"
            
            total_loss = torch.tensor(0.0, device=train_device)
            with torch.autocast(device_type=train_device, dtype=torch.bfloat16):
                for s in trange(gradient_accumulation_steps, desc="Micro-batches", leave=False):
                    start_idx = s * micro_batch_size
                    end_idx = (s + 1) * micro_batch_size
                    input_ids_mb = input_ids[start_idx:end_idx]
                    labels_mb = labels[start_idx:end_idx]
                    logits = llm(input_ids_mb).logits # type: ignore
                    vocab_size = logits.size(-1)
                    assert logits.shape == (micro_batch_size, seq_len, vocab_size), f"logits shape: {logits.shape}"
                    loss = F.cross_entropy(logits.view(-1, vocab_size), labels_mb.view(-1)) / gradient_accumulation_steps
                    loss.backward()
                    total_loss += loss.detach()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=1.0) # type: ignore
            
            current_lr = lr_scheduler.get_last_lr()[0]
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                tpar.set_description(f"Loss: {total_loss.item():.4f}")
                print_and_log(f"Epoch {epoch+1} Iter {batch_idx+1}/{len(loader)}, Loss: {total_loss.item():.4f}, Grad Norm: {grad_norm.item():.4f}, LR: {current_lr:.6f}")
            
            if (batch_idx + 1) % eval_interval == 0:
                print_and_log(f"--- Evaluation at Epoch {epoch+1} Iteration {batch_idx+1} ---")
                evaluate_model_on_dataset(llm, val_loader, train_device)
    
    print_and_log("Final evaluation after training completion:")
    evaluate_model_on_dataset(llm, val_loader, train_device)
    
    if not is_test:
        # Save the fine-tuned model
        llm.save_pretrained(output_dir) # type: ignore
        print_and_log(f"Fine-tuned model saved to {output_dir}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_and_log(f"Training completed in {elapsed_time/60:.2f} minutes.")


if __name__ == "__main__":
   typer.run(main) 
