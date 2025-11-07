# TrainForge

**TrainForge** is an easy-to-use framework for training Large Language Models (LLMs) and Vision-Language Models (VLMs). It provides simple YAML configuration files to control training with Unsloth, supports multi-GPU setup, and leverages Unsloth's optimizations for efficient training.

## Features

- **Multi-GPU Training**: Seamless distributed training across multiple GPUs with optimized NCCL configuration
- **LLM & VLM Support**: Unified framework for both language and vision-language model fine-tuning
- **LoRA Optimization**: Efficient parameter-efficient fine-tuning with configurable LoRA adapters
- **Production-Ready Scripts**: One-command training scripts for common use cases
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Integrated Logging**: Built-in support for Weights & Biases (wandb) and comprehensive training metrics
- **Memory Efficient**: Support for 4-bit and 8-bit quantization with gradient checkpointing

## Performance Achievements

TrainForge has successfully fine-tuned **Qwen 3 4B** models with the following performance metrics:

### Training Configuration
- **Model**: Qwen 3 4B
- **Method**: LoRA fine-tuning
- **Epochs**: 3
- **Dataset Size**: 14,5K samples
- **Training Time**: ~12 hours
- **VRAM Usage**: 13 GB

### Training Parameters
```yaml
max_seq_length: 4096
per_device_train_batch_size: 20
gradient_accumulation_steps: 1
```

These results demonstrate TrainForge's efficiency in training large models with optimal resource utilization.

## 📋 Requirements

- **Python**: 3.10 or 3.11
- **CUDA**: Compatible NVIDIA GPUs with CUDA support
- **Conda**: Recommended for environment management (optional but recommended)
- **Hardware**: Multi-GPU setup recommended for distributed training

## Installation

### 1. Environment Setup

We recommend using Conda for environment management:

```bash
# Create and activate conda environment
conda create -n unsloth python=3.10 -y
conda activate unsloth
```

### 2. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt
```

**Note**: If you encounter issues with `accelerate`, install it separately:
```bash
pip install accelerate --no-deps --no-build-isolation
```

### 3. Verify Installation

```bash
# Check GPU availability
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Quick Start

### LLM Supervised Fine-Tuning

#### Using Training Scripts (Recommended)

**Multi-GPU Training:**
```bash
# Train on GPUs 1 and 2
./scripts/train_unsloth_llm.sh 2 "1,2"
```

**Single GPU Training:**
```bash
# Train on GPU 1
./scripts/train_unsloth_llm.sh 1 "1"
```

#### Manual Training with Accelerate

```bash
# Multi-GPU example
CUDA_VISIBLE_DEVICES="1,2" accelerate launch --multi-gpu --num_processes 2 examples/train.py
```

### VLM Supervised Fine-Tuning

#### Using Training Scripts (Recommended)

**Single GPU Example:**
```bash
./scripts/train_unsloth_vlm.sh 1 "0"
```

**Multi-GPU Example:**
```bash
# Adjust first argument for number of GPUs, second for device IDs
./scripts/train_unsloth_vlm.sh 2 "0,1"
```

#### Manual Training with Accelerate

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 examples/unsloth/vlm/examples_sft.py
```

## Configuration

TrainForge uses YAML configuration files for flexible and reproducible training setups.

### Configuration Structure

- **LLM Configs**: `config/unsloth/llm/sft.yaml`
- **VLM Configs**: `config/unsloth/vlm/sft.yaml`

### Key Configuration Parameters

#### Model Configuration
```yaml
model:
  model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"  # or local path
  max_seq_length: 4096
  load_in_4bit: true
  load_in_8bit: false
  trust_remote_code: false
```

#### Training Hyperparameters
```yaml
hyperparams:
  per_device_train_batch_size: 20
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
  num_train_epochs: 3
  warmup_steps: 100
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"
  bf16: true
  fp16: false
```

#### LoRA Configuration
```yaml
lora:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

#### Dataset Configuration
```yaml
dataset:
  dataset_name: "ChaosAiVision/VI_CoT-RAG"
  dataset_split: "train"
  test_size: 0.1
  shuffle: true
  text_field: "text"
```

#### Logging Configuration
```yaml
hyperparams:
  report_to: "wandb"  # Set to "wandb" for Weights & Biases integration
  logging_steps: 10
  output_dir: "output_unsloth_llm_sft"
```

## Project Structure

```
TrainForge/
├── config/                 # YAML configuration files
│   └── unsloth/
│       ├── llm/
│       └── vlm/
├── examples/               # Example training scripts
│   ├── train.py
│   └── unsloth/
│       ├── llm/
│       └── vlm/
├── scripts/                # Training shell scripts
│   ├── train_unsloth_llm.sh
│   └── train_unsloth_vlm.sh
├── src/                    # Core framework code
│   └── forge/
│       ├── core/           # Base configurations and utilities
│       ├── module/         # Training modules (Unsloth, HuggingFace)
│       └── utils/          # Helper utilities
└── requirements.txt        # Python dependencies
```

## 🔧 Advanced Usage

### Using Local Model Paths

For models that Unsloth cannot auto-download, download the model locally and reference it:

```yaml
model:
  model_name_or_path: "/data/models/MyModel"
  max_seq_length: 4096
  load_in_4bit: false
  load_in_8bit: false
  trust_remote_code: false
```

**Download model locally:**
```bash
huggingface-cli download <org/model> --local-dir /data/models/MyModel --local-dir-use-symlinks False
```

**Note**: Ensure sufficient disk space and that Git LFS is installed for large model files.

### LoRA and Quantization Compatibility

Some models do not support training with 4-bit/8-bit quantization when using LoRA. If you encounter shape/size mismatch errors (e.g., "mismatch size" in weight tensors), disable quantization:

```yaml
model:
  load_in_4bit: false
  load_in_8bit: false
```

### Multi-GPU Training Notes

- **First startup**: Initial training run may take 30-120 seconds while `accelerate/transformers` initialize
- **DDP Configuration**: The trainer automatically sets `ddp_find_unused_parameters=False` to avoid DDP unused-parameter issues with LoRA
- **NCCL Settings**: Training scripts configure NCCL for optimal multi-GPU performance

## Outputs & Logging

### Checkpoints

Training checkpoints are saved to the configured `output_dir` (e.g., `output_unsloth_llm_sft/`). Checkpoints include:
- Model weights and LoRA adapters
- Training state and optimizer states
- Configuration snapshots

### Weights & Biases Integration

When `report_to: "wandb"` is set in your configuration, training metrics are automatically logged to Weights & Biases:
- Loss curves (training and validation)
- Learning rate schedules
- GPU utilization
- Training speed metrics

## Roadmap

### Completed
- Multi-GPU training scripts for LLM SFT
- Multi-GPU training scripts for VLM SFT
- YAML-based configuration system
- LoRA fine-tuning support
- Weights & Biases integration

### In Development
- Reinforcement Learning (RL) training scripts: Support for RLHF, DPO, PPO, and other RL-based fine-tuning methods
- FP4 quantization for Blackwell architecture: Optimized quantization support for NVIDIA's Blackwell GPU architecture

## Troubleshooting

### Common Issues

**Issue**: DDP unused parameter warnings
- **Solution**: The framework automatically handles this. Ensure `ddp_find_unused_parameters=False` is set (default).

**Issue**: Shape/size mismatch errors with quantization
- **Solution**: Disable quantization (`load_in_4bit: false`, `load_in_8bit: false`) when using LoRA with incompatible models.

**Issue**: Model download failures
- **Solution**: Download models locally using `huggingface-cli` and reference the local path in configuration.

**Issue**: NCCL communication errors in multi-GPU training
- **Solution**: Training scripts configure NCCL settings automatically. For custom setups, ensure proper network interface configuration.

## License

This project is available under a **Dual License** model:

### GNU General Public License v3.0 (GPL-3.0)
- **Free for**: Research, academic, and non-commercial use
- **Terms**: See [LICENSE](LICENSE) file for full GPL-3.0 terms
- **Rights**: You can use, modify, and distribute under GPL-3.0 terms

### Commercial License
- **Required for**: Production, commercial, and enterprise use
- **Benefits**: 
  - No requirement to open-source your modifications
  - Commercial support available
  - Priority bug fixes and feature requests
- **Details**: See [LICENSE](LICENSE) file for commercial license terms
- **Contact**: For commercial licensing inquiries, please contact us via [LinkedIn](https://www.linkedin.com/in/nhattruongnguyen20022003/) or visit our [GitHub repository](https://github.com/ChaosAIVision)

**Summary**: Use GPL-3.0 for research and non-commercial purposes. For production or commercial use, a commercial license is required.




## Contact
Linkedin
https://www.linkedin.com/in/nhattruongnguyen20022003/

---

