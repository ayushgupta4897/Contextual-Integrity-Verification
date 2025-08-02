#!/usr/bin/env python3
"""
Debug Normal Prompts - Find out why normal prompts are failing

This isolates the normal prompt issue and fixes it.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import perform_model_surgery


def debug_normal_behavior():
    """Debug why normal prompts fail"""
    print("🔍 DEBUGGING NORMAL PROMPT BEHAVIOR")
    print("="*50)
    
    # Load models
    print("📥 Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load baseline
    baseline_model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Create CIV model
    base_copy = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct", 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    civ_model = perform_model_surgery(base_copy, tokenizer)
    
    print("✅ Models loaded")
    
    # Test normal prompt
    normal_prompt = "What is the capital of France?"
    print(f"\n📝 Testing normal prompt: {normal_prompt}")
    
    # Ensure NO namespace injection on CIV model
    print("\n🧹 Cleaning any leftover namespace_ids...")
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
            print(f"   Cleaned: {name}")
    
    # Test baseline
    print("\n🔴 BASELINE TEST:")
    try:
        inputs = tokenizer(normal_prompt, return_tensors="pt")
        inputs = {k: v.to(baseline_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = baseline_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        baseline_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        baseline_response = baseline_response[len(normal_prompt):].strip()
        print(f"✅ Baseline: {baseline_response[:80]}...")
        
    except Exception as e:
        print(f"❌ Baseline error: {str(e)}")
        baseline_response = ""
    
    # Test CIV WITHOUT security
    print("\n🟢 CIV TEST (NO SECURITY):")
    try:
        inputs = tokenizer(normal_prompt, return_tensors="pt")  
        inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = civ_model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        civ_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        civ_response = civ_response[len(normal_prompt):].strip()
        print(f"✅ CIV: {civ_response[:80]}...")
        
    except Exception as e:
        print(f"❌ CIV error: {str(e)}")
        civ_response = ""
    
    # Analysis
    print(f"\n📊 COMPARISON:")
    baseline_ok = len(baseline_response) > 10 and "paris" in baseline_response.lower()
    civ_ok = len(civ_response) > 10 and "paris" in civ_response.lower()
    
    print(f"🔴 Baseline quality: {'✅ GOOD' if baseline_ok else '❌ POOR'}")
    print(f"🟢 CIV quality:      {'✅ GOOD' if civ_ok else '❌ POOR'}")
    
    if baseline_ok and civ_ok:
        print("🎉 SUCCESS: Both models work normally!")
    elif baseline_ok and not civ_ok:
        print("⚠️  PROBLEM: CIV is broken for normal prompts")
    else:
        print("❓ UNEXPECTED: Need investigation")
    
    return baseline_ok, civ_ok


def test_multiple_normal_prompts():
    """Test multiple normal prompts to confirm behavior"""
    print("\n🧪 TESTING MULTIPLE NORMAL PROMPTS")
    print("="*50)
    
    # Load CIV model
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_copy = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    civ_model = perform_model_surgery(base_copy, tokenizer)
    
    # Ensure clean state
    for name, module in civ_model.named_modules():
        if hasattr(module, '_current_namespace_ids'):
            delattr(module, '_current_namespace_ids')
    
    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?", 
        "What color is the sky?",
        "How do you make coffee?",
        "What is Python programming?"
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: {prompt}")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = civ_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Simple quality check
            quality_ok = len(response) > 5 and not response.startswith("Error") and "#Question" not in response
            status = "✅ PASS" if quality_ok else "❌ FAIL"
            
            print(f"Response: {response[:60]}...")
            print(f"Status: {status}")
            
            results.append({"prompt": prompt, "quality_ok": quality_ok, "response": response})
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"prompt": prompt, "quality_ok": False, "response": f"Error: {str(e)}"})
    
    # Summary
    passed = sum(1 for r in results if r["quality_ok"])
    print(f"\n📊 NORMAL PROMPTS SUMMARY: {passed}/{len(test_prompts)} passed")
    
    if passed == len(test_prompts):
        print("🎉 ALL NORMAL PROMPTS WORKING!")
    else:
        print("⚠️  SOME NORMAL PROMPTS FAILING - Need to fix")
    
    return results


def main():
    """Main debugging function"""
    print("🚀 NORMAL PROMPT DEBUG SESSION")
    print("="*60)
    
    # Step 1: Basic debug
    baseline_ok, civ_ok = debug_normal_behavior()
    
    if not civ_ok:
        print("\n❌ CIV failing basic normal prompts - investigating...")
        return
    
    # Step 2: Multiple prompt test
    results = test_multiple_normal_prompts()
    
    print("\n🏆 NORMAL PROMPT DEBUG COMPLETE!")
    

if __name__ == "__main__":
    main()