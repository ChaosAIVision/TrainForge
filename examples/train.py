
#  CUDA_VISIBLE_DEVICES="0,1" accelerate launch --multi-gpu --num_processes 2 train.py


import datasets
from datasets import load_dataset, DatasetDict
from unsloth.chat_templates import get_chat_template
import wandb
import os 
from unsloth import FastLanguageModel

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

accelerator = Accelerator()
rank_idx = accelerator.process_index
print(f"[PID {os.getpid()}, Rank {rank_idx}] Accelerator initialized. Distributed: {accelerator.distributed_type}, Device: {accelerator.device}, Num_processes: {accelerator.num_processes}", flush=True)

import torch
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset', 
    job_type="training", 
    anonymous="allow"
)
MAX_SEQ_LENGTH= 4096

# 2. Load tokenizer and model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    # model_name='/home/chaos/Documents/chaos/repo/save_memory_llm_training/model_qwen_v3_1.7B_math_template',
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    full_finetuning = False, # We have full finetuning now!
    load_in_8bit = False,
    load_in_4bit = True,
    device_map=  accelerator.device
)


SYSTEM_PROMPT= """
Bạn là một AI Assistant chuyên nghiệp. Và output bạn trả về luôn luôn là JSON theo đúng template quy định.

"""

INSTRUCTION_PROMPT = """
Bạn là một mô hình chuyên gia, nhiệm vụ của bạn là tạo ra output JSON hoàn chỉnh theo cấu trúc qui định.

Input được cung cấp dưới dạng một object JSON có khóa chính là "context", giá trị là:

#### Context Json
{{context}}

Yêu cầu:
1. Đọc kỹ toàn bộ nội dung trong "context".
2. Thực hiện các bước tư duy để:
   - Hiểu đúng yêu cầu (understand_requirements).
   - Lập kế hoạch trả lời chi tiết theo từng bước (plan).
   - Trích xuất hoặc suy ra các từ khóa thể hiện câu trả lời đúng (successful_keywords).
   - Viết phần phản hồi chính (response) bằng ngôn ngữ chuyên môn, súc tích, không tính cảm.
   - Chọn đáp án đúng trong multiple_choices (nếu có) và ghi vào key "choice".
   - Nếu cần dùng công cụ ngoài, liệt kê vào "tools" (hoặc để [] nếu không cần).

3. Hãy phản hồi theo định dạng JSON sau:
{{
    "understand_requirements": "Liệt kê vài dòng để hiểu từ ngữ cảnh và prompt xem người dùng họ muốn gì",
    "plan": "Liệt kê quy trình cần thực hiện, hiểu từ ngữ cảnh, prompts để đưa ra những bước hành động",
    "successful_keywords": "Tóm tắt những thông tin sẽ có trong đáp án khi trả lời để thành công",
    "response": "Trả lời với users ở đây như chat bot, theo format từ plan và successful key words",
    "choice": "Trả về thêm đáp án là ABCD hoặc là Yes No nếu trong multiple choice có đưa đáp án và yêu cầu, nếu không để trống",
    "tools": "Nếu có yêu cầu thực hiện như 1 agent thì trả về mã agent chạy tại đây nếu không để trống"
}}
"""

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "qwen-3", # works with qwen-2.5 but not qwen-3
# )

# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Instruct')


model = FastLanguageModel.get_peft_model(
    model,
    r = 8,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0.1, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ,
)



def generate_conversation(examples):
    # conversations = examples['messages']
    # for conversation in conversations:
    #     for role in conversation:
    #         if role['role'] == 'system':
    #                 role['content'] = SYSTEM_PROMPT
    
    prompts_json_list = examples['prompts_json']
    response_json_list = examples['response_json']
    conversations = []
    for prompt_json, response_json in zip(prompts_json_list, response_json_list):
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt_json
            },
            {
                "role": "assistant",
                "content": response_json
            }
        ]
        conversations.append(conversation)





    conversations =  tokenizer.apply_chat_template(conversations, batched = True, tokenize = False, add_generation_prompt = False )
    conversations = [
    conversation + tokenizer.eos_token
    for conversation in conversations
    
]

    return { "text": conversations }

dataset = load_dataset('ChaosAiVision/medical_1k_json', split ='train')
split_dataset = dataset.train_test_split(test_size=0.1 , shuffle=True, seed=42)

dataset_train = split_dataset['train']
dataset_validation = split_dataset['test']

dataset_train = dataset_train.map(generate_conversation, batched = True)
dataset_validation = dataset_validation.map(generate_conversation, batched = True)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# model = FastLanguageModel.for_training(model= model)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    eval_dataset= dataset_validation,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_gpu_eval_batch_size= 1 , 
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 5000,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        warmup_ratio= 0.05,
        weight_decay = 0,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "output_test_train_sft",
        report_to = "wandb", 
        save_strategy="steps",
        save_steps= 100,
        save_total_limit=3,  
        eval_strategy= 'steps',
        eval_steps= 100,
        do_eval= True,
        metric_for_best_model= 'eval_loss',
        logging_strategy= 'steps',
        ddp_find_unused_parameters = False

    ),
)
# trainer.train()

from unsloth import unsloth_train
# unsloth_train fixes gradient_accumulation_steps
# trainer_stats = trainer.train()
trainer_stats = unsloth_train(trainer)


accelerator.wait_for_everyone()
if accelerator.is_main_process:
    print(f"[PID {os.getpid()}, Rank {rank_idx}] Training complete. Saving final model...", flush=True)
    # Save logic here
