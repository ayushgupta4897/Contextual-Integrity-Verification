#!/usr/bin/env python3
"""
Test Working CIV Model - 20 Layer Partial Surgery

Test the working CIV implementation that replaces only the first 20 layers,
which maintains normal functionality while providing architectural security.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import perform_model_surgery


def test_working_civ_normal_prompts():
    """Test normal prompts with the working 20-layer CIV model"""
    print("🧪 TESTING WORKING CIV MODEL - NORMAL PROMPTS")
    print("="*60)
    
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
    
    print("🔧 Creating WORKING CIV model (20 layers)...")
    base_copy = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    civ_model = perform_model_surgery(base_copy, tokenizer, max_layers=20)
    
    # Ensure clean state - NO security active
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
    print("🧹 CIV model cleaned - NO security should be active")
    
    # Test prompts
    test_prompts = [
        "What is the capital of Japan?",
        "Who invented the telephone?",
        "What is 2 + 2?",
        "Write a short poem about cats",
        "How do you make coffee?",
        "What color is the sky?",
        "Explain gravity briefly",
        "Hello, how are you?",
        "What is Python programming?",
        "Tell me a joke",
    ]
    
    print(f"\n🚀 Testing {len(test_prompts)} normal prompts...")
    
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
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            civ_response = tokenizer.decode(civ_outputs[0], skip_special_tokens=True)
            civ_response = civ_response[len(prompt):].strip()
            
            # Quality assessment
            has_repetitive_question = "QuestionQuestion" in civ_response[:50]
            has_content = len(civ_response.strip()) > 3
            reasonable_start = not civ_response.startswith(("#", ".", "?", "!"))
            diversity = len(set(civ_response[:30])) > 5  # Some character diversity
            
            is_good = has_content and not has_repetitive_question and reasonable_start and diversity
            status = "✅ GOOD" if is_good else "❌ POOR"
            
            print(f"CIV Response: {civ_response[:80]}...")
            print(f"Quality: {status}")
            
            if is_good:
                passed += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Results
    success_rate = passed / len(test_prompts) * 100
    print(f"\n📊 WORKING CIV NORMAL PROMPTS RESULTS")
    print("="*50)
    print(f"✅ Good responses: {passed}/{len(test_prompts)} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT: Working CIV model handles normal prompts well!")
    elif success_rate >= 60:
        print("✅ GOOD: Working CIV model is functional for normal use")
    elif success_rate >= 40:
        print("⚠️  FAIR: Working CIV model has some issues but is usable")
    else:
        print("❌ POOR: Working CIV model still needs fixes")
    
    return success_rate


def compare_baseline_vs_working_civ():
    """Compare baseline vs working CIV for quality"""
    print(f"\n🔍 BASELINE VS WORKING CIV COMPARISON")
    print("="*50)
    
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
    civ_model = perform_model_surgery(base_copy, tokenizer, max_layers=20)
    
    # Clean CIV model
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
    
    # Test prompts
    comparison_prompts = [
        "What is artificial intelligence?",
        "Explain how photosynthesis works",
        "Write a story about a brave knight",
    ]
    
    for prompt in comparison_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Baseline
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(baseline_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            baseline_outputs = baseline_model.generate(
                **inputs, max_new_tokens=25, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        baseline_response = tokenizer.decode(baseline_outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        
        # CIV
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            civ_outputs = civ_model.generate(
                **inputs, max_new_tokens=25, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        civ_response = tokenizer.decode(civ_outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        
        print(f"🔴 Baseline: {baseline_response[:80]}...")
        print(f"🟢 CIV (20L): {civ_response[:80]}...")
        
        # Simple quality comparison
        baseline_good = len(baseline_response.strip()) > 10 and "Question" not in baseline_response[:20]
        civ_good = len(civ_response.strip()) > 10 and "Question" not in civ_response[:20]
        
        print(f"📊 Quality: Baseline {'✅' if baseline_good else '❌'} | CIV {'✅' if civ_good else '❌'}")


def main():
    """Test the working CIV model"""
    print("🚀 TESTING WORKING CIV MODEL (20-LAYER PARTIAL SURGERY)")
    print("="*80)
    print("🎯 Goal: Verify CIV works for normal prompts after discovering the breaking point")
    
    # Test normal prompts
    success_rate = test_working_civ_normal_prompts()
    
    # Comparison
    compare_baseline_vs_working_civ()
    
    print(f"\n🏁 WORKING CIV TEST COMPLETE!")
    print(f"🎯 Normal Prompt Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 70:
        print("🎉 BREAKTHROUGH ACHIEVED!")
        print("✅ CIV model now works for normal prompts")
        print("🔄 Next: Test if security features still work with 20 layers")
    else:
        print("⚠️  Still needs optimization, but major progress made")
    
    return success_rate >= 70


if __name__ == "__main__":
    main()