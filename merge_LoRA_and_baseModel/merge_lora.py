#!/usr/bin/env python3
"""
Merge a LoRA adapter with a base model into a single complete model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16
):
    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ Merged model successfully saved to {output_path}")
    print(f"  - Model files: {output_path}")
    print(f"  - Tokenizer files: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--base-model", type=str, required=True, 
                       help="Path to base model (HuggingFace format)")
    parser.add_argument("--lora-adapter", type=str, required=True,
                       help="Path to LoRA adapter directory (with adapter_config.json)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path for merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Data type for the merged model")
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    
    merge_lora_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output,
        torch_dtype=dtype_map[args.dtype]
    )
