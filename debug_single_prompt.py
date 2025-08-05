#!/usr/bin/env python3

"""
Debug single normal prompt to identify the exact regression
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from civ_core import CIVProtectedModel, create_namespace_ids

def test_single_normal_prompt():
    """Test a single normal prompt to see exact output"""
    
    print("🔍 DEBUGGING SINGLE NORMAL PROMPT")
    print("=" * 50)
    
    # Load model and tokenizer
    print("Loading model...")
    model_path = "models/llama-3.2-3b-instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="mps",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create CIV protected model
    print("Creating CIV protected model...")
    civ_model = CIVProtectedModel(model, tokenizer, max_layers=20)
    
    # Test prompt
    test_prompt = "What is 15 + 25?"
    
    print(f"\n🧪 Testing prompt: '{test_prompt}'")
    print("-" * 40)
    
    # Baseline response
    print("🔍 BASELINE MODEL:")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        baseline_output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    baseline_response = tokenizer.decode(baseline_output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"BASELINE: {baseline_response}")
    
    # CIV response with explicit namespace IDs
    print("\n🛡️ CIV MODEL:")
    namespace_ids = create_namespace_ids(
        user_text=test_prompt,
        tokenizer=tokenizer
    )
    
    # Convert to tensor
    namespace_tensor = torch.tensor([namespace_ids], dtype=torch.long).to(model.device)
    
    with torch.no_grad():
        civ_output = civ_model.generate_with_civ(
            input_ids=inputs['input_ids'],
            namespace_ids=namespace_tensor,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    civ_response = tokenizer.decode(civ_output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"CIV:      {civ_response}")
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"Baseline length: {len(baseline_response)}")
    print(f"CIV length: {len(civ_response)}")
    print(f"First 50 chars of CIV: '{civ_response[:50]}'")
    
    if len(set(civ_response.split())) < 5:
        print("❌ DETECTED: Repetitive/garbled output")
    else:
        print("✅ Output appears normal")

if __name__ == "__main__":
    test_single_normal_prompt()