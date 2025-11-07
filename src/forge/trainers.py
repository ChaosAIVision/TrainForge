import yaml
from typing import Optional, Union, Dict, Any
from dataclasses import asdict
from pathlib import Path

from .core.llm.base_config import ModelConfig as LLMModelConfig, HyperParamsConfig as LLMHyperParamsConfig, LayerConfig as LLMLayerConfig, LoraConfig as LLMLoraConfig, DatasetConfig as LLMDatasetConfig
from .core.vlm.base_config import ModelConfig as VLMModelConfig, HyperParamsConfig as VLMHyperParamsConfig, LayerConfig as VLMLayerConfig, LoraConfig as VLMLoraConfig, DatasetConfig as VLMDatasetConfig
from .module.unsloth.llm.sft.train import UnslothLLMSFTTrainer
from .module.unsloth.vlm.sft.train import UnslothVLMSFTTrainer
from .module.huggingface.llm.sft.train import HuggingFaceLLMSFTTrainer
from .module.huggingface.vlm.sft.train import HuggingFaceVLMSFTTrainer


class TrainerFactory:
    """Factory class for creating trainers based on configuration."""
    
    @staticmethod
    def _load_config_from_yaml(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @staticmethod
    def _create_llm_configs_from_dict(config_dict: Dict[str, Any]) -> tuple:
        """Create LLM config objects from dictionary."""
        model_config = LLMModelConfig(**config_dict['model'])
        hyperparams_config = LLMHyperParamsConfig(**config_dict['hyperparams'])
        layer_config = LLMLayerConfig(**config_dict['layer'])
        dataset_config = LLMDatasetConfig(**config_dict['dataset'])
        
        lora_config = None
        if 'lora' in config_dict and config_dict['lora'] is not None:
            lora_config = LLMLoraConfig(**config_dict['lora'])
        
        return model_config, hyperparams_config, layer_config, dataset_config, lora_config
    
    @staticmethod
    def _create_vlm_configs_from_dict(config_dict: Dict[str, Any]) -> tuple:
        """Create VLM config objects from dictionary."""
        model_config = VLMModelConfig(**config_dict['model'])
        hyperparams_config = VLMHyperParamsConfig(**config_dict['hyperparams'])
        layer_config = VLMLayerConfig(**config_dict['layer'])
        dataset_config = VLMDatasetConfig(**config_dict['dataset'])
        
        lora_config = None
        if 'lora' in config_dict and config_dict['lora'] is not None:
            lora_config = VLMLoraConfig(**config_dict['lora'])
        
        return model_config, hyperparams_config, layer_config, dataset_config, lora_config


class UnslothLLMTrainer:
    """User-friendly Unsloth LLM Trainer interface."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize trainer with config file path or config dictionary."""
        if isinstance(config, str):
            config_dict = TrainerFactory._load_config_from_yaml(config)
        else:
            config_dict = config
        
        (self.model_config, self.hyperparams_config, self.layer_config, 
         self.dataset_config, self.lora_config) = TrainerFactory._create_llm_configs_from_dict(config_dict)
        
        self.trainer = UnslothLLMSFTTrainer(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            layer_config=self.layer_config,
            dataset_config=self.dataset_config,
            lora_config=self.lora_config
        )
    
    def train(self) -> Dict[str, Any]:
        """Start training."""
        return self.trainer.train()


class UnslothVLMTrainer:
    """User-friendly Unsloth VLM Trainer interface."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize VLM trainer with config file path or config dictionary."""
        if isinstance(config, str):
            config_dict = TrainerFactory._load_config_from_yaml(config)
        else:
            config_dict = config
        
        (self.model_config, self.hyperparams_config, self.layer_config, 
         self.dataset_config, self.lora_config) = TrainerFactory._create_vlm_configs_from_dict(config_dict)
        
        self.trainer = UnslothVLMSFTTrainer(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            layer_config=self.layer_config,
            dataset_config=self.dataset_config,
            lora_config=self.lora_config
        )
    
    def train(self) -> Dict[str, Any]:
        """Start VLM training."""
        return self.trainer.train()


class HuggingFaceLLMTrainer:
    """User-friendly HuggingFace LLM Trainer interface."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize HF LLM trainer with config file path or config dictionary."""
        if isinstance(config, str):
            config_dict = TrainerFactory._load_config_from_yaml(config)
        else:
            config_dict = config
        
        (self.model_config, self.hyperparams_config, self.layer_config, 
         self.dataset_config, self.lora_config) = TrainerFactory._create_llm_configs_from_dict(config_dict)
        
        self.trainer = HuggingFaceLLMSFTTrainer(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            layer_config=self.layer_config,
            dataset_config=self.dataset_config,
            lora_config=self.lora_config
        )
    
    def train(self) -> Dict[str, Any]:
        """Start HF LLM training."""
        return self.trainer.train()


class HuggingFaceVLMTrainer:
    """User-friendly HuggingFace VLM Trainer interface."""
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """Initialize HF VLM trainer with config file path or config dictionary."""
        if isinstance(config, str):
            config_dict = TrainerFactory._load_config_from_yaml(config)
        else:
            config_dict = config
        
        (self.model_config, self.hyperparams_config, self.layer_config, 
         self.dataset_config, self.lora_config) = TrainerFactory._create_vlm_configs_from_dict(config_dict)
        
        self.trainer = HuggingFaceVLMSFTTrainer(
            model_config=self.model_config,
            hyperparams_config=self.hyperparams_config,
            layer_config=self.layer_config,
            dataset_config=self.dataset_config,
            lora_config=self.lora_config
        )
    
    def train(self) -> Dict[str, Any]:
        """Start HF VLM training."""
        return self.trainer.train()