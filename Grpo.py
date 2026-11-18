!pip install -U transformers datasets peft trl accelerate

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

dataset = Dataset.from_dict({
    "prompt": [
        "What is 2 + 2?",
        "What is 3 + 3?"
    ]
})

def reward_function(completions, **kwargs):
    rewards = []
    answers = ["4", "6"]

    for completion, answer in zip(completions, answers):
        text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else str(completion)
        )
        rewards.append(1.0 if answer in text else 0.0)

    return rewards

args = GRPOConfig(
    output_dir="./grpo-model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    num_generations=2,
    logging_steps=1,
    report_to="none"
)

trainer = GRPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    reward_funcs=reward_function,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
)

trainer.train()
trainer.save_model("./grpo-model")
