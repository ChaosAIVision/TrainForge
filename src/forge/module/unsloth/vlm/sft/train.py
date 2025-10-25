

from typing import Optional, Tuple, Any, Dict
from datasets import Dataset
from transformers import TrainingArguments
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import FastVisionModel, is_bfloat16_supported, unsloth_train
from accelerate import Accelerator
import os

from forge.core.vlm.base_config import ModelConfig, HyperParamsConfig, LayerConfig, LoraConfig, DatasetConfig
from forge.core.vlm.loader import BaseModelLoader, LoaderFactory


class UnslothVLMModelLoader(BaseModelLoader):
    """Unsloth model loader for VLM."""
    def load_model_and_tokenizer(self, config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load VLM model and tokenizer using Unsloth."""
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=config.model_name_or_path,
            max_seq_length=config.max_seq_length,
            dtype=config.dtype,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map
        )
        return model, tokenizer


class UnslothVLMSFTTrainer:
    """Unsloth VLM SFT Trainer with production-ready architecture."""
    
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
        """Load and setup VLM model and tokenizer."""
        model_loader = UnslothVLMModelLoader()
        # Ensure device_map is set per-process BEFORE loading the model
        if self.model_config.device_map is None:
            self.model_config.device_map = self.accelerator.device
        self.model, self.tokenizer = model_loader.load_model_and_tokenizer(self.model_config)
        
        if self.lora_config is not None:
            self.model = FastVisionModel.get_peft_model(
                self.model,
                finetune_vision_layers=self.layer_config.finetune_vision_layers,
                finetune_language_layers=self.layer_config.finetune_language_layers,
                finetune_attention_modules=self.layer_config.finetune_attention_modules,
                finetune_mlp_modules=self.layer_config.finetune_mlp_modules,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                random_state=self.lora_config.random_state,
                use_rslora=self.lora_config.use_rslora,
                loftq_config=self.lora_config.loftq_config,
                target_modules=self.lora_config.target_modules,
            )
    
    def setup_dataset(self):
        """Setup the VLM dataset for training."""
        dataset_loader = LoaderFactory.create_dataset_loader("huggingface")
        train_dataset, eval_dataset = dataset_loader.load_dataset(self.dataset_config)
        
        # Convert datasets to conversation format using list comprehension
        converted_train_dataset = [self.convert_to_conversation(sample) for sample in train_dataset]
        converted_eval_dataset = [self.convert_to_conversation(sample) for sample in eval_dataset] if eval_dataset else None
        
        self.train_dataset = converted_train_dataset
        self.eval_dataset = converted_eval_dataset


        # # Create new datasets from converted data
        # self.train_dataset = Dataset.from_list(converted_train_dataset)
        # self.eval_dataset = Dataset.from_list(converted_eval_dataset) if converted_eval_dataset else None
    
    def convert_to_conversation(self, sample):
        """Convert a single sample to conversation format for VLM training."""
        # Define template for question formatting
        TEMPLATE = "Câu hỏi: {question}"
        
        # Use instruction prompt from dataset if available, otherwise use Vietnamese default
        instruction_prompt = sample.get("instruction_prompt", "Bạn là một trợ lý AI thông minh, chuyên trả lời câu hỏi về hình ảnh bằng tiếng Việt chuẩn, rõ ràng và mạch lạc.")
        if instruction_prompt is None:
            instruction_prompt = "Bạn là một trợ lý AI thông minh, chuyên trả lời câu hỏi về hình ảnh bằng tiếng Việt chuẩn, rõ ràng và mạch lạc."
        
        conversation = [
            {"role": "system", "content": instruction_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEMPLATE.format(question=sample["question"])},
                    {"type": "image", "image": sample["image"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["response"]}
                ]
            },
        ]
        return {"messages": conversation}
    
    def setup_trainer(self):
        """Setup VLM SFT trainer with configurations."""
        training_args = SFTConfig(
            per_device_train_batch_size=self.hyperparams_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.hyperparams_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.hyperparams_config.gradient_accumulation_steps,
            warmup_steps=self.hyperparams_config.warmup_steps,
            num_train_epochs=self.hyperparams_config.num_train_epochs,
            max_steps=self.hyperparams_config.max_steps if self.hyperparams_config.max_steps is not None else -1,
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
            logging_strategy="steps",
            ddp_find_unused_parameters=False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 4096,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator = UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=self.train_dataset,
            # eval_dataset=self.eval_dataset,
            max_seq_length=self.model_config.max_seq_length,
            dataset_num_proc=self.dataset_config.dataset_num_proc,
            packing=self.dataset_config.packing,
            args=training_args,
        )
    
    def train(self) -> Dict[str, Any]:
        """Execute VLM training pipeline."""
        # Log accelerator setup similar to LLM
        rank_idx = self.accelerator.process_index
        print(f"[PID {os.getpid()}, Rank {rank_idx}] Accelerator initialized. Distributed: {self.accelerator.distributed_type}, Device: {self.accelerator.device}, Num_processes: {self.accelerator.num_processes}", flush=True)

        self.setup_model()
        self.setup_dataset() 
        self.setup_trainer()
        
        trainer_stats = unsloth_train(self.trainer)
        
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("VLM Training completed successfully")
            
        return trainer_stats

#  CUDA_VISIBLE_DEVICES="0" accelerate launch - --num_processes 1 /home/chaos/Documents/chaos/repo/TrainForge/examples/unsloth/vlm/examples_sft.py
