import json
import os
import yaml
import typing
from pydantic import BaseModel
from typing import Callable, Literal


class SftConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    num_epochs: int = 5
    learning_rate: float = 5e-5
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    eval_interval: int = -1
    max_examples: int = 1024
    compile_model: bool = False
    clip_gradients: float = 1.0

class RLConfig(BaseModel):
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    cliprange: float = 0.2


class ExpertIterConfig(BaseModel):
    question_batch_size: int = 512
    sample_batch_size: int = 8
    n_ei_steps: int = 5
    use_all_positive: bool = False
    do_rollout: bool = True


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from a file and return as a dictionary"""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path) as f:
        if ext == ".json":
            return json.load(f)
        elif ext in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
