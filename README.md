# TrainForge — Quick Start Guide

This guide shows how to install dependencies and run LLM/VLM fine-tuning with multi‑GPU using the provided scripts or manual commands. Keep it simple, and verify GPUs are available before starting.

## Requirements
- Python `3.10` or `3.11`
- Optional: Conda (recommended)
- GPUs for multi‑GPU; CPU-only works for quick tests

## 1. Setup Environment
Recommended with Conda:
- `conda create -n unsloth python=3.10 -y`
- `conda activate unsloth`

Install repo dependencies:
- `pip install -r requirements.txt`

Tips:
- Avoid building PyTorch from source. Use the official wheel channels above.
- If you install `accelerate`, prefer: `pip install accelerate --no-deps --no-build-isolation`

## 2. Verify GPUs
- `nvidia-smi`
- `python -c "import torch; print(torch.cuda.device_count())"`

## 3. Run LLM SFT Training
Recommended (script):
- Multi‑GPU on GPUs 1 and 2: 
  - `./scripts/train_unsloth_llm.sh 2 "1,2"`
- Single GPU (GPU 1):
  - `./scripts/train_unsloth_llm.sh 1 "1"`

Manual (accelerate):
- `CUDA_VISIBLE_DEVICES="1,2" accelerate launch --multi-gpu --num_processes 2 examples/train.py`

Notes:
- First startup can take 30–120 seconds while `accelerate/transformers` initialize.
- The trainer config sets `ddp_find_unused_parameters=False` to avoid DDP unused‑param issues with LoRA.

## 4. Run VLM SFT Training
Recommended (script):
- Single GPU example:
  - `./scripts/train_unsloth_vlm.sh 1 "0"`
- Adjust the first argument for number of GPUs and the second for visible device IDs.

Manual (accelerate):
- Example pattern:
  - `CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 examples/unsloth/vlm/examples_sft.py`

## 5. Configuration
- LLM configs under `config/unsloth/llm/`
- VLM configs under `config/unsloth/vlm/`
- Key parameters:
  - `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`
  - `max_seq_length`, `num_train_epochs`, `output_dir`
  - Logging/reporting: set `report_to` (e.g. `"wandb"`)

## 6. Outputs & Logging
- Checkpoints and logs go to the configured `output_dir` (e.g. `output_unsloth_llm_sft/`).
- If `report_to="wandb"`, runs will appear in your Weights & Biases project.

## 7. Model Compatibility Notes
- LoRA + quantization: Some models do not support training with 4‑bit/8‑bit quantization. Set `load_in_4bit: false` and `load_in_8bit: false` when fine‑tuning with LoRA to avoid shape/size mismatch errors (e.g., “mismatch size” in weight tensors).
- Use local model path: For models that Unsloth cannot auto‑download, download the model to disk and point `model_name_or_path` to the local directory instead of a Hugging Face name.
- Example YAML:
```
model:
  model_name_or_path: "/data/models/MyModel"
  max_seq_length: 4096
  load_in_4bit: false
  load_in_8bit: false
  trust_remote_code: false
```
- Download locally:
  - `huggingface-cli download <org/model> --local-dir /data/models/MyModel --local-dir-use-symlinks False`
  - Ensure enough disk space and that Git LFS is installed for large files.

## Troubleshooting
- Long startup: wait ~1–2 minutes; this is normal for multi‑GPU init.
- NCCL/port conflicts: ensure no other training jobs are using the same GPUs; free memory via `nvidia-smi`.
- OOM (out of memory): reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps`.
- Torch install errors: confirm your CUDA version and use the matching wheel index URL.

## Optional: torchrun
If you prefer `torchrun` over `accelerate`:
- `CUDA_VISIBLE_DEVICES="1,2" torchrun --nproc_per_node=2 examples/train.py`

That’s it. Start with the scripts, confirm GPUs are recognized, and iterate on configs in `config/unsloth/*` to fit your dataset and hardware.
