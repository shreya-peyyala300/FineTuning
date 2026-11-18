!pip install -U transformers datasets peft trl accelerate

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

dataset = Dataset.from_dict({
    "query": ["What is 2 + 2?"]
})

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1
)

# PPO APIs differ across TRL versions.
# This is the classic PPO workflow, not guaranteed to run with latest TRL.
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset
)

query = tokenizer("What is 2 + 2?", return_tensors="pt").input_ids[0]
response = model.generate(query.unsqueeze(0), max_new_tokens=20)[0][len(query):]

# Example reward only; use a trained reward model in real RLHF.
reward = __import__("torch").tensor(1.0)

ppo_trainer.step([query], [response], [reward])
