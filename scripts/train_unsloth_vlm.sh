#!/bin/bash

# TrainForge - Simple Unsloth LLM Training Script
# Usage: ./train_unsloth_llm.sh [num_gpus] [gpu_ids]
# Example: ./train_unsloth_llm.sh 2 "1,2"

# Default values
NUM_GPUS=${1:-1}
GPU_IDS=${2:-"0"}

# Get script directory correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting TrainForge LLM Training..."
echo "Project root: $PROJECT_ROOT"
echo "Number of GPUs: $NUM_GPUS"
echo "GPU IDs: $GPU_IDS"

# Change to project directory
cd "$PROJECT_ROOT"

# Export environment
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
# Reduce NCCL hangs on single host
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=lo

# Activate conda environment robustly
eval "$(conda shell.bash hook)"
conda activate unsloth || { echo "Conda activate failed."; exit 1; }

# Do not auto-install deps here; ensure environment prepared beforehand
# python -c "import accelerate" 2>/dev/null || pip install --no-deps accelerate

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running single GPU training..."
    python -u examples/unsloth/vlm/example_sft.py
else
    echo "Running multi-GPU training..."
    accelerate launch --multi-gpu --num_processes "$NUM_GPUS" --num_machines 1 --main_process_port 29501 --mixed_precision bf16 examples/unsloth/llm/example_sft.py
fi

echo "✅ Training completed!"