import json
import os
import yaml
import typing
from pydantic import BaseModel


class SftConfig(BaseModel):
    model_id: str = "Qwen/Qwen2.5-Math-1.5B"
    learning_rate: float = 1e-5
    batch_size: int = 128
    gradient_accumulation_steps: int = 8
    num_epochs: int = 10
    eval_interval: int = -1


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
