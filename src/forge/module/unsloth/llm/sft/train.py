from unsloth import FastLanguageModel, is_bfloat16_supported, unsloth_train
from typing import Optional, Tuple, Any, Dict
from datasets import Dataset
from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTTrainer
from accelerate import Accelerator

from forge.core.llm.base_config import ModelConfig, HyperParamsConfig, LayerConfig, LoraConfig, DatasetConfig
from forge.core.llm.loader import BaseModelLoader, LoaderFactory


class UnslothModelLoader(BaseModelLoader):
    """Unsloth model loader for LLM."""
    
    def load_model_and_tokenizer(self, config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer using Unsloth."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name_or_path,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map
        )
        return model, tokenizer


class UnslothLLMSFTTrainer:
    """Unsloth LLM SFT Trainer with production-ready architecture."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        hyperparams_config: HyperParamsConfig,
        layer_config: LayerConfig,
        dataset_config: DatasetConfig,
        lora_config: Optional[LoraConfig] = None,
        accelerator: Optional[Accelerator] = None
    ):
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config
        self.layer_config = layer_config
        self.dataset_config = dataset_config
        self.lora_config = lora_config
        self.accelerator = accelerator or Accelerator()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup_model(self):
        """Load and setup model and tokenizer."""
        model_loader = UnslothModelLoader()
        self.model, self.tokenizer = model_loader.load_model_and_tokenizer(self.model_config)
        
        if self.model_config.device_map is None:
            self.model_config.device_map = self.accelerator.device
        
        if self.lora_config is not None:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.lora_config.r,
                target_modules=self.layer_config.target_modules,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                use_gradient_checkpointing=self.lora_config.use_gradient_checkpointing,
                random_state=self.lora_config.random_state,
                use_rslora=self.lora_config.use_rslora,
                loftq_config=self.lora_config.loftq_config,
            )
    
    def setup_dataset(self):
        """Load and setup dataset."""
        dataset_loader = LoaderFactory.create_dataset_loader("huggingface")
        self.train_dataset, self.eval_dataset = dataset_loader.load_dataset(self.dataset_config)
        self.train_dataset = self.train_dataset.map(self.formatting_prompts_func, batched=True)
        self.eval_dataset = self.eval_dataset.map(self.formatting_prompts_func, batched=True)
    

    def formatting_prompts_func(self, examples):
        convos = examples["messages"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts }

    def setup_trainer(self):
        """Setup SFT trainer with configurations."""
        training_args = TrainingArguments(
            per_device_train_batch_size=self.hyperparams_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.hyperparams_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.hyperparams_config.gradient_accumulation_steps,
            warmup_steps=self.hyperparams_config.warmup_steps,
            num_train_epochs=self.hyperparams_config.num_train_epochs,
            learning_rate=self.hyperparams_config.learning_rate,
            fp16=not is_bfloat16_supported() if not hasattr(self.hyperparams_config, 'fp16') else self.hyperparams_config.fp16,
            bf16=is_bfloat16_supported() if not hasattr(self.hyperparams_config, 'bf16') else self.hyperparams_config.bf16,
            logging_steps=self.hyperparams_config.logging_steps,
            optim=self.hyperparams_config.optim,
            weight_decay=self.hyperparams_config.weight_decay,
            lr_scheduler_type=self.hyperparams_config.lr_scheduler_type,
            seed=self.hyperparams_config.seed,
            output_dir=self.hyperparams_config.output_dir,
            warmup_ratio=self.hyperparams_config.warmup_ratio,
            save_steps=self.hyperparams_config.save_steps,
            save_total_limit=self.hyperparams_config.save_total_limit,
            eval_steps=self.hyperparams_config.eval_steps,
            metric_for_best_model=self.hyperparams_config.metric_for_best_model,
            dataloader_num_workers=self.hyperparams_config.dataloader_num_workers,
            remove_unused_columns=self.hyperparams_config.remove_unused_columns,
            report_to=self.hyperparams_config.report_to,
            save_strategy="steps",
            eval_strategy="steps" if self.eval_dataset else "no",
            logging_strategy="steps"
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            dataset_text_field= "text",
            max_seq_length=self.model_config.max_seq_length,
            dataset_num_proc=self.dataset_config.dataset_num_proc,
            packing=self.dataset_config.packing,
            args=training_args,
        )
    
    def train(self) -> Dict[str, Any]:
        """Execute training pipeline."""
        self.setup_model()
        self.setup_dataset() 
        self.setup_trainer()
        
        trainer_stats = unsloth_train(self.trainer)
        
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("Training completed successfully")
            
        return trainer_stats