!pip install -U transformers datasets peft trl bitsandbytes accelerate

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = Dataset.from_dict({
    "prompt": ["What is 2 + 2?"],
    "chosen": ["2 + 2 equals 4."],
    "rejected": ["2 + 2 equals 5."]
})

args = DPOConfig(
    output_dir="./dpo-model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    report_to="none"
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
)

trainer.train()
trainer.save_model("./dpo-model")
