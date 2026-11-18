!pip install -U transformers datasets peft accelerate bitsandbytes trl

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./qlora-hr-model"

# Tesla T4: use FP16, not BF16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model.config.use_cache = False

# Disable gradient checkpointing for this initial experiment
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False,
)

print("Model loaded successfully")



lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)



dataset = load_dataset(
    "json",
    data_files="/content/train.jsonl",
    split="train",
)

print("Training examples:", len(dataset))
print(dataset[0])




training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    gradient_checkpointing=False,
    optim="adamw_torch",
    report_to="none",
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
)

# Check the data type of every trainable parameter

# Keep the quantized base model unchanged.
# Convert only trainable adapter parameters to FP32.
for name, param in trainer.model.named_parameters():
    if param.requires_grad and param.dtype == torch.bfloat16:
        param.data = param.data.float()

print("Trainable parameter data types:")
for name, param in trainer.model.named_parameters():
    if param.requires_grad:
        print(name, param.dtype)

trainer.train()


trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("QLoRA fine-tuning completed!")
print("Adapter saved at:", OUTPUT_DIR)
