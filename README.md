# 🚀 TrainForge - Easy AI Training Framework

<div align="center">

![TrainForge Logo](https://img.shields.io/badge/TrainForge-AI%20Training%20Framework-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSI+PHBhdGggZD0iTTEyIDJMNiA3VjE3TDEyIDIyTDE4IDE3VjdMMTIgMloiIHN0cm9rZT0iIzMwODRGRiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)

**A professional framework for easily training AI models (LLM & VLM)**

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Examples](#-examples) • [Configuration](#-configuration)

</div>

---

## 🎯 What is TrainForge?

TrainForge is a Python framework that helps you train AI models simply and efficiently. You just need to:

1. **Create a YAML configuration file** - No complex code needed
2. **Run 3 lines of code** - The framework automatically handles everything
3. **Monitor results** - Integrated with Weights & Biases

**Suitable for:** Students, researchers, AI engineers who want to train models quickly.

---

## ✨ Features

| 🎯 **Cross-platform** | Supports both **Unsloth** (fast) and **HuggingFace** (stable) |
|---|---|
| 🤖 **Multi-model** | Trains both **LLM** (language models) and **VLM** (vision-language models) |
| ⚡ **Optimization** | Automatic memory and training speed optimization |
| 🔧 **Simple configuration** | Just a YAML file, no complex code needed |
| 🚀 **Multi-GPU** | Automatically distributes training across multiple GPUs |
| 📊 **Experiment tracking** | Integrated with Weights & Biases |

---

## 🚀 Quick Start

### Step 1: Installation
```bash
# Activate conda environment (if using unsloth)
conda activate unsloth

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create a configuration file
Create `my_config.yaml`:

```yaml
model:
  model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"
  max_seq_length: 4096
  load_in_4bit: true

hyperparams:
  per_device_train_batch_size: 1
  learning_rate: 0.0001
  num_train_epochs: 3
  output_dir: "my_model_output"

dataset:
  dataset_name: "ChaosAiVision/medical_1k_json"
  test_size: 0.1
  text_field: "text"
```

### Step 3: Run training
```python
from forge import UnslothLLMTrainer

# Initialize trainer
trainer = UnslothLLMTrainer(config="my_config.yaml")

# Start training
trainer.train()
```

**Or run directly:**
```bash
conda activate unsloth
PYTHONPATH=$PYTHONPATH:./src python examples/unsloth/llm/example_sft.py
```

---

## 📦 Installation

### System requirements
- Python 3.8+
- NVIDIA GPU (recommended)
- CUDA 11.8+ or 12.0+

### Install from source
```bash
# Clone repository
git clone https://github.com/yourusername/TrainForge.git
cd TrainForge

# Install dependencies
pip install -r requirements.txt

# Check installation
python -c "from src.forge import UnslothLLMTrainer; print('✅ Installation successful!')"
```

### Install Unsloth environment (recommended)
```bash
# Create a new conda environment
conda create --name unsloth python=3.11 -y
conda activate unsloth

# Install unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 💻 Usage

### 1. Train LLM (Language Model)

```python
from forge import UnslothLLMTrainer

# Use existing configuration file
trainer = UnslothLLMTrainer(config="config/unsloth/llm/sft.yaml")
results = trainer.train()

print(f"Training completed! Results: {results}")
```

### 2. Train VLM (Vision-Language Model)

```python
from forge import UnslothVLMTrainer

# Train a model that can "see" and "speak"
trainer = UnslothVLMTrainer(config="config/unsloth/vlm/sft.yaml")
results = trainer.train()
```

### 3. Multi-GPU training

```bash
# Use 2 GPUs
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi-gpu --num_processes 2 train.py
```

---

## 📚 Examples

### Example 1: Train a medical chatbot
```python
"""
Train a model to answer medical questions
"""
from forge import UnslothLLMTrainer

def main():
    print("🏥 Training medical chatbot...")
    
    # Use existing medical dataset
    trainer = UnslothLLMTrainer(config="config/unsloth/llm/sft.yaml")
    results = trainer.train()
    
    print("✅ Training completed!")
    print(f"📊 Results: {results}")

if __name__ == "__main__":
    main()
```

### Example 2: Train an image understanding model
```python
"""
Train a model that can describe images
"""
from forge import UnslothVLMTrainer

def main():
    print("👁️ Training vision model...")
    
    trainer = UnslothVLMTrainer(config="config/unsloth/vlm/sft.yaml")
    results = trainer.train()
    
    print("✅ Model has learned to 'see' images!")

if __name__ == "__main__":
    main()
```

---

## ⚙️ Configuration

### Configuration file structure

Each YAML file has 5 main sections:

```yaml
# 1. Model configuration
model:
  model_name_or_path: "unsloth/Llama-3.2-3B-Instruct"  # Model name
  max_seq_length: 4096                                   # Max length
  load_in_4bit: true                                     # Memory saving

# 2. Training parameters
hyperparams:
  per_device_train_batch_size: 1    # Batch size
  learning_rate: 0.0001             # Learning rate
  num_train_epochs: 3               # Number of epochs
  output_dir: "my_output"           # Output directory

# 3. Layer configuration
layer:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  full_finetuning: false            # Partial training only

# 4. Dataset configuration
dataset:
  dataset_name: "ChaosAiVision/medical_1k_json"  # Dataset name
  test_size: 0.1                                  # Test ratio
  text_field: "text"                              # Text field

# 5. LoRA configuration (optional)
lora:
  r: 8                              # LoRA rank
  lora_alpha: 32                    # Alpha parameter
  lora_dropout: 0.1                 # Dropout rate
```

### Available configuration files

| File | Description | Model type |
|------|-------------|------------|
| `config/unsloth/llm/sft.yaml` | Train LLM with Unsloth | LLM |
| `config/unsloth/vlm/sft.yaml` | Train VLM with Unsloth | VLM |

### Important parameters

#### Memory optimization
- `load_in_4bit: true` - Reduces memory by 75%
- `per_device_train_batch_size: 1` - Small batch size
- `gradient_accumulation_steps: 4` - Accumulate gradients

#### Speed optimization
- `packing: true` - Pack sequences (5x faster)
- `bf16: true` - Use bfloat16
- `dataloader_num_workers: 2` - Multi-threaded data loading

---

## 🛠️ Project Structure
