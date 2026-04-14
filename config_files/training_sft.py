import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# =========================
# 1. Load dataset
# =========================
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

raw_data = load_jsonl("sft_candidates.json")

dataset = Dataset.from_list(raw_data)

# =========================
# 2. Load tokenizer + model
# =========================
model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # IMPORTANT

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,   #updated (no torch_dtype warning)
    device_map="auto"
)

# =========================
# 3. Apply LoRA (same as before)
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
# 4. SFT Config
# =========================
training_args = SFTConfig(
    output_dir="./qwen-sft-sql",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,  # 🔥 higher LR for LoRA
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    max_length=1024,
    fp16=True,

    # 🔥 IMPORTANT for chat datasets
    assistant_only_loss=True,
)

# =========================
# 5. Trainer
# =========================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,   #new API (like DPO fix)
    peft_config=peft_config,      # optional (you already wrapped, but OK)
)

# =========================
# 6. Train
# =========================
trainer.train()

# =========================
# 7. Save LoRA adapter
# =========================
model.save_pretrained("./qwen-sft-sql-lora")
tokenizer.save_pretrained("./qwen-sft-sql-lora")