from cs336_alignment.logger import print_and_log
import torch
from transformers import PreTrainedModel
from cs336_alignment.config import LoraParaConfig
from peft import LoraConfig, get_peft_model


def apply_lora(model: PreTrainedModel, lora_config: LoraParaConfig) -> torch.nn.Module:
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    trainable, total = 0, 0
    for n, p in peft_model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print_and_log(
        f"Trainable params: {trainable/1e6:.2f}M | Total: {total/1e6:.2f}M | Ratio: {100*trainable/total:.4f}%"
    )

    # Collect layers where LoRA is applied and layers without LoRA
    applied_lora_types = {}  # layer_type -> count
    layers_without_lora = {}  # layer_type -> param_count

    # Helper function to extract layer type (remove layer numbers and base_model prefix)
    def get_layer_type(layer_path: str) -> str:
        # Remove base_model.model prefix if present
        path = layer_path.replace("base_model.model.", "")
        # Remove .base_layer suffix (added by PEFT for original weights)
        path = path.replace(".base_layer", "")
        # Remove layer numbers like "layers.0.", "layers.19." etc.
        import re

        path = re.sub(r"\.layers\.\d+\.", ".layers.X.", path)
        path = re.sub(r"^layers\.\d+\.", "layers.X.", path)
        return path

    # First pass: find all layers with LoRA
    for n, p in peft_model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            # Extract the base layer path (before .lora_A or .lora_B)
            layer_path = n.rsplit(".lora_", 1)[0]
            layer_type = get_layer_type(layer_path)
            applied_lora_types[layer_type] = applied_lora_types.get(layer_type, 0) + 1

    # Second pass: find all layers without LoRA
    for n, p in peft_model.named_parameters():
        # Skip LoRA parameters themselves
        if "lora_A" in n or "lora_B" in n:
            continue

        # Get the layer path (remove .weight, .bias, etc.)
        if ".weight" in n or ".bias" in n:
            layer_path = n.rsplit(".", 1)[0]
        else:
            layer_path = n

        layer_type = get_layer_type(layer_path)

        # Check if this layer type has LoRA applied
        has_lora = layer_type in applied_lora_types

        if not has_lora:
            if layer_type not in layers_without_lora:
                layers_without_lora[layer_type] = 0
            layers_without_lora[layer_type] += p.numel()

    print_and_log(f"\nLoRA applied layer types ({len(applied_lora_types)} types):")
    for layer_type in sorted(applied_lora_types.keys()):
        count = applied_lora_types[layer_type]
        print_and_log(f"  ✓ {layer_type} (x{count})")

    total_params_without_lora = sum(layers_without_lora.values())
    ratio_without_lora = 100 * total_params_without_lora / total if total > 0 else 0
    print_and_log(
        f"\nLayer types WITHOUT LoRA ({len(layers_without_lora)} types, {total_params_without_lora/1e6:.2f}M params, {ratio_without_lora:.2f}% of total):"
    )
    for layer_type in sorted(layers_without_lora.keys()):
        param_count = layers_without_lora[layer_type]
        param_ratio = 100 * param_count / total if total > 0 else 0
        print_and_log(
            f"  ✗ {layer_type}: {param_count/1e6:.4f}M params ({param_ratio:.4f}%)"
        )

    # Estimate memory usage for training with bfloat16 and AdamW optimizer
    print_and_log("\n=== Memory Estimation (bfloat16 + AdamW) ===")

    # Model memory: base model (bfloat16) + trainable params (bfloat16)
    base_model_memory_gb = total * 2 / 1e9  # 2 bytes per param (bfloat16)
    trainable_memory_gb = trainable * 2 / 1e9  # 2 bytes per trainable param

    # AdamW optimizer states: momentum and variance (float32 for both)
    optimizer_memory_gb = (
        trainable * 4 * 2 / 1e9
    )  # 4 bytes * 2 states (momentum + variance)

    # Gradients (same dtype as trainable params, bfloat16)
    gradient_memory_gb = trainable * 2 / 1e9  # 2 bytes per trainable param

    total_memory_gb = (
        base_model_memory_gb
        + trainable_memory_gb
        + optimizer_memory_gb
        + gradient_memory_gb
    )

    print_and_log(f"Model (base + trainable): {base_model_memory_gb:.2f} GB")
    print_and_log(f"Gradients (bfloat16): {gradient_memory_gb:.2f} GB")
    print_and_log(f"AdamW states (float32): {optimizer_memory_gb:.2f} GB")
    print_and_log(f"Total estimated: {total_memory_gb:.2f} GB")
    print_and_log(f"Note: Activation memory depends on batch size and sequence length")

    return peft_model
