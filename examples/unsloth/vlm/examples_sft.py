"""
Simple example for Unsloth VLM SFT Training using TrainForge framework.
"""

from forge import UnslothVLMTrainer


def main():
    """Basic VLM training using YAML configuration."""
    print("=== TrainForge VLM Training Example ===")
    
    # Simple usage - just load config and train
    trainer = UnslothVLMTrainer(config="config/unsloth/vlm/sft.yaml")
    trainer_stats = trainer.train()
    
    print(f"Training completed! Stats: {trainer_stats}")


if __name__ == "__main__":
    main()