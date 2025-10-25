
import unsloth
from forge import UnslothLLMTrainer

trainer = UnslothLLMTrainer(config="config/unsloth/llm/sft.yaml")
trainer.train()
