#!/usr/bin/env python3
"""
Test Fixed CIV Model with RoPE - Quick Validation

This tests if adding Rotary Position Embedding (RoPE) fixes the 
repetitive "QuestionQuestion..." pattern issue.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import perform_model_surgery


def test_fixed_civ_model():
    """Test the fixed CIV model with RoPE"""
    print("🧪 TESTING FIXED CIV MODEL WITH ROPE")
    print("="*50)
    print("Goal: Verify RoPE fixes the repetitive pattern issue")
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("📥 Loading baseline model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("🔧 Creating FIXED CIV model with RoPE...")
    base_copy = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    civ_model = perform_model_surgery(base_copy, tokenizer)
    
    # Ensure clean state - NO security active
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
    print("🧹 CIV model cleaned - NO security should be active")
    
    # Test prompts - quick validation
    test_prompts = [
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?", 
        "What color is the sky?",
        "How do you make coffee?",
        "What is Python programming?", 
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain gravity in simple terms",
    ]
    
    print(f"\n🚀 Testing {len(test_prompts)} prompts...")
    
    passed = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        
        try:
            # Test CIV model
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                civ_outputs = civ_model.generate(
                    **inputs,
                    max_new_tokens=15,  # Shorter for quick test
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            civ_response = tokenizer.decode(civ_outputs[0], skip_special_tokens=True)
            civ_response = civ_response[len(prompt):].strip()
            
            # Check for improvement
            has_repetitive_pattern = "QuestionQuestion" in civ_response[:50]
            has_content = len(civ_response.strip()) > 3
            starts_reasonably = not civ_response.startswith(("#", "Question", "The#"))
            
            is_good = has_content and not has_repetitive_pattern and starts_reasonably
            status = "✅ GOOD" if is_good else "❌ POOR"
            
            print(f"CIV Response: {civ_response[:60]}...")
            print(f"Quality: {status}")
            
            if is_good:
                passed += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Results
    success_rate = passed / len(test_prompts) * 100
    print(f"\n📊 QUICK TEST RESULTS")
    print("="*30)
    print(f"✅ Good responses: {passed}/{len(test_prompts)} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🎉 EXCELLENT: RoPE fix appears to work!")
    elif success_rate >= 50:
        print("✅ GOOD: Significant improvement with RoPE")
    elif success_rate >= 25:
        print("⚠️  FAIR: Some improvement, needs more work")
    else:
        print("❌ POOR: Still broken, RoPE didn't fix it")
    
    return success_rate


def compare_before_after():
    """Quick comparison to show the improvement"""
    print(f"\n🔍 BASELINE VS FIXED CIV COMPARISON")
    print("="*40)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    baseline_model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    base_copy = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    civ_model = perform_model_surgery(base_copy, tokenizer)
    
    # Clean CIV model
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
    
    # Test prompt
    prompt = "What is the meaning of life?"
    
    print(f"\n📝 Prompt: {prompt}")
    
    # Baseline
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(baseline_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        baseline_outputs = baseline_model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    baseline_response = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    # CIV
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        civ_outputs = civ_model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    civ_response = tokenizer.decode(civ_outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    
    print(f"🔴 Baseline: {baseline_response[:80]}...")
    print(f"🟢 CIV (Fixed): {civ_response[:80]}...")
    
    # Analysis
    civ_has_repetitive = "QuestionQuestion" in civ_response
    print(f"📊 CIV Repetitive Pattern: {'❌ YES (Still broken)' if civ_has_repetitive else '✅ NO (Fixed!)'}")


def main():
    """Run the fixed CIV test"""
    print("🚀 TESTING CIV MODEL WITH ROPE FIX")
    print("="*60)
    
    # Main test
    success_rate = test_fixed_civ_model()
    
    # Comparison
    compare_before_after()
    
    print(f"\n🏁 ROPE FIX TEST COMPLETE!")
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🎉 CIV MODEL IS NOW WORKING FOR NORMAL PROMPTS!")
        print("✅ Ready to test security features next")
    else:
        print("⚠️  CIV model still needs more fixes")


if __name__ == "__main__":
    main()