#Das ist unsere erste funktionierende config-file, mit der das Training von Qwen direkt ueber Huggingface geklappt hat.
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# =========================
# 1. Load dataset
# =========================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

raw_data = load_json("data.json")

# =========================
# 2. Format prompt (Qwen chat format)
# =========================
def format_prompt(messages):
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"<|system|>\n{msg['content']}\n"
        elif msg["role"] == "user":
            text += f"<|user|>\n{msg['content']}\n"
    text += "<|assistant|>\n"
    return text

def transform(example):
    return {
        "prompt": format_prompt(example["messages"]),
        "chosen": example["chosen"]["content"],
        "rejected": example["rejected"]["content"],
    }

dataset = Dataset.from_list([transform(x) for x in raw_data])

# =========================
# 3. Load tokenizer + model
# =========================
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =========================
# 4. Apply LoRA (IMPORTANT)
# =========================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# =========================
# 5. DPO Config
# =========================
training_args = DPOConfig(
    output_dir="./qwen-dpo-sql",
    per_device_train_batch_size=2,  # gerne etwas hoch (2), vorher 1
    gradient_accumulation_steps=8,  # auch mal etwas hoch probieren, 8 (man rechnet dann Gradient x Batch-Size = effektive Batchgröße), vorher 4
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    max_length=1024,
    fp16=True,
)

# =========================
# 6. Trainer
# =========================
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class =tokenizer,
)

# =========================
# 7. Train
# =========================
trainer.train()

# =========================
# 8. Save LoRA adapter
# =========================
model.save_pretrained("./qwen-dpo-sql-lora")
tokenizer.save_pretrained("./qwen-dpo-sql-lora")