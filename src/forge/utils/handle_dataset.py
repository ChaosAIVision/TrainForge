from typing import List , Dict , Optional

def unsloth_llm_format(examples):
    
    """
    args:
        tokenizer: (PreTrainedTokenizer): tokenizer for apply_chat_template
    inputs:
        examples: (dict): item from huggingface dataset format
    outputs:
        text: (str): A List of dictionary with key "text" and value is apply_chat_template of conversation
    """

    texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in examples["messages"]]
    return {"text": texts}

from transformers.tokenization_utils import PreTrainedTokenizer


def unsloth_vlm_format(examples):
    """
    Format the examples for VLM training inputs.

    args:
        examples: (dict): A dictionary containing the examples with keys "question", "image", "response", 
                         and optionally "instruction_prompt".
    
    outputs:
        dict: A dictionary containing the formatted examples with key "messages".

    This function converts dataset samples to conversation format for VLM training.
    Uses instruction_prompt from dataset if available, otherwise uses default.
    """
    
    def convert_to_conversation(sample):
        # Use instruction prompt from dataset if available, otherwise use default
        instruction_prompt = sample.get("instruction_prompt", "You are a helpful assistant")
        
        # Simple template - just use the question directly
        user_text = sample["question"]
        
        conversation = [
            {"role": "system", "content": instruction_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
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
        return conversation
    
    # Convert all examples to conversation format
    conversations = []
    for i in range(len(examples["question"])):
        sample = {
            "question": examples["question"][i],
            "image": examples["image"][i],
            "response": examples["response"][i]
        }
        
        # Add instruction_prompt if it exists in the dataset
        if "instruction_prompt" in examples:
            sample["instruction_prompt"] = examples["instruction_prompt"][i]
        
        conversations.append(convert_to_conversation(sample))
    
    return {"messages": conversations}