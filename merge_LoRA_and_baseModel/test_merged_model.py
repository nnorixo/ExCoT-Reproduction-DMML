# test_merged_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/pfad/zum/gemergeten/modell"

print(f"Loading merged model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test-Prompt
prompt = "Generate a SQL query for: List all users from Berlin"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
