from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

"""
This script showcase how I fine-tuned mistral-7b. 
Reference from: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl
"""

dataset = load_dataset("json", data_files="train_dataset.json", split="train")

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
model, tokenizer = setup_chat_format(model, tokenizer)

args = TrainingArguments(
    output_dir="mistral-7b-audio-tags",  # Directory to save and repository id
    num_train_epochs=3,                  # Number of training epochs
    per_device_train_batch_size=3,       # Batch size per device during training
    gradient_accumulation_steps=2,       # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,         # Use gradient checkpointing to save memory
    optim="adamw_torch_fused",           # Use fused adamw optimizer
    logging_steps=10,                    # Log every 10 steps
    save_strategy="epoch",               # Save checkpoint every epoch
    learning_rate=2e-4,                  # Learning rate, based on QLoRA paper
    bf16=True,                           # Use bfloat16 precision
    tf32=True,                           # use tf32 precision
    max_grad_norm=0.3,                   # Max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                   # Warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",        
    push_to_hub=False,                  
    report_to="wandb",                   
    run_name="mistral-7b-audio-tags",  
) 

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

max_seq_length = 3072 

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  
        "append_concat_token": False, 
    }
)

trainer.train()
trainer.save_model()