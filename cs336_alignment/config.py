import json
import os
import yaml
import typing
from pydantic import BaseModel


class SftConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    num_epochs: int = 20
    learning_rate: float = 1e-5
    micro_batch_size: int = 128
    gradient_accumulation_steps: int = 8
    eval_interval: int = -1
    max_examples: int = 1024
    compile_model: bool = False


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
