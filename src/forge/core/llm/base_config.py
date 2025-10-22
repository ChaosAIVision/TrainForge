from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for LLM model loading during training."""
    model_name_or_path: str
    max_seq_length: int = 4096
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    device_map: Optional[str] = None

@dataclass
class HyperParamsConfig:
    """Configuration for training hyperparameters."""
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    optim: str = "adamw_8bit"
    seed: int = 3407
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    report_to: str = "wandb"
    output_dir: str = "output"

@dataclass
class LayerConfig:
    """Configuration to specify which layers will be trained."""
    target_modules: List[str]
    full_finetuning: bool = False

@dataclass 
class LoraConfig:
    """Configuration for LoRA (Low-Rank Adaptation) layer settings."""
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None

@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    dataset_name: str
    dataset_split: str = "train"
    test_size: float = 0.1
    shuffle: bool = True
    seed: int = 42
    text_field: str = "text"
    dataset_num_proc: int = 2
    packing: bool = False


