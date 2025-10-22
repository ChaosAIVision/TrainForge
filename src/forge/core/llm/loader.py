from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedModel

from .base_config import ModelConfig, DatasetConfig


class BaseModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def load_model_and_tokenizer(self, config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer based on configuration."""
        pass


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_dataset(self, config: DatasetConfig) -> Tuple[Dataset, Dataset]:
        """Load train and validation datasets based on configuration."""
        pass


class HuggingFaceDatasetLoader(BaseDatasetLoader):
    """HuggingFace dataset loader implementation."""
    
    def load_dataset(self, config: DatasetConfig) -> Tuple[Dataset, Dataset]:
        """Load dataset from HuggingFace Hub."""
        dataset = load_dataset(config.dataset_name, split=config.dataset_split)
        
        if config.test_size > 0:
            split_dataset = dataset.train_test_split(
                test_size=config.test_size,
                shuffle=config.shuffle,
                seed=config.seed
            )
            train_dataset = split_dataset['train']
            val_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            val_dataset = None
            
        return train_dataset, val_dataset


class LoaderFactory:
    """Factory class for creating loaders."""
    
    @staticmethod
    def create_dataset_loader(loader_type: str = "huggingface") -> BaseDatasetLoader:
        """Create dataset loader based on type."""
        if loader_type == "huggingface":
            return HuggingFaceDatasetLoader()
        else:
            raise ValueError(f"Unknown dataset loader type: {loader_type}")


