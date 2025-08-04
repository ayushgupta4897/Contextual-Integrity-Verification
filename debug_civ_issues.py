#!/usr/bin/env python3

"""
Quick diagnostic test for CIV issues
Based on judge LLM recommendations
"""

import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_core import CIVProtectedModel, create_namespace_ids, TrustLevel

def test_gradient_leak_probe():
    """Test if gradients leak through forbidden attention paths"""
    print("🔍 TESTING GRADIENT LEAK PROBE")
    print("=" * 50)
    
    try:
        # Load model
        tokenizer = AutoTokenizer.from_pretrained("models/llama-3.2-3b-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "models/llama-3.2-3b-instruct",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        protected_model = CIVProtectedModel(base_model, tokenizer, max_layers=1)
        
        # Get correct hidden size from model config
        hidden_size = base_model.config.hidden_size
        print(f"Model hidden size: {hidden_size}")
        
        # Create test scenario: WEB -> SYSTEM -> WEB (WEB should not attend to SYSTEM)
        hidden_states = torch.randn(1, 3, hidden_size, requires_grad=True)  # [batch, seq_len, hidden_size]
        namespace_ids = torch.tensor([[20, 100, 20]])  # WEB, SYSTEM, WEB
        
        # Get the CIV attention layer
        civ_attn = protected_model.base_model.model.layers[0].self_attn
        
        # Forward pass through attention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Call attention with namespace_ids
            output = civ_attn(
                hidden_states=hidden_states,
                namespace_ids=namespace_ids
            )
            
            # If output is tuple, take first element
            if isinstance(output, tuple):
                output = output[0]
        
        # Backward pass - compute gradients for first token (WEB)
        output[0, 0].sum().backward()
        
        # Check gradient magnitude for first token (should be minimal if properly masked)
        web_token_grad = hidden_states.grad[0, 0, :].abs().sum().item()
        
        print(f"WEB token gradient magnitude: {web_token_grad:.6f}")
        print(f"Expected: ~0 (if gradient-safe masking works)")
        
        if web_token_grad < 1e-6:
            print("✅ Gradient leak test PASSED - no significant gradients")
            return True
        else:
            print("❌ Gradient leak test FAILED - gradients still flowing")
            return False
            
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_attention_masking():
    """Test if basic attention masking is working"""
    print("\n🔍 TESTING BASIC ATTENTION MASKING")
    print("=" * 50)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("models/llama-3.2-3b-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "models/llama-3.2-3b-instruct",
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        protected_model = CIVProtectedModel(base_model, tokenizer, max_layers=1)
        
        # Simple test input
        input_text = "Hello world"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Create simple namespace: USER tokens
        namespace_ids = torch.tensor([[80] * input_ids.shape[1]])  # All USER trust
        
        print(f"Input: {input_text}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Namespace IDs shape: {namespace_ids.shape}")
        
        # Generate response
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            outputs = protected_model.generate_with_civ(
                input_ids=input_ids,
                namespace_ids=namespace_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        generated_tokens = outputs[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Response: '{response}'")
        
        # Check if response is reasonable (not garbled)
        if len(response.strip()) > 0 and not all(token == response.strip().split()[0] for token in response.strip().split()):
            print("✅ Basic masking test PASSED - response looks reasonable")
            return True
        else:
            print("❌ Basic masking test FAILED - response is garbled or empty")
            return False
            
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_trust_constraints():
    """Test simple trust constraint logic"""
    print("\n🔍 TESTING TRUST CONSTRAINT LOGIC")
    print("=" * 50)
    
    try:
        from civ_core import TrustLevel
        
        # Test trust hierarchy
        system_trust = TrustLevel.SYSTEM.value  # 100
        user_trust = TrustLevel.USER.value      # 80  
        web_trust = TrustLevel.WEB.value        # 20
        
        print(f"SYSTEM trust: {system_trust}")
        print(f"USER trust: {user_trust}")
        print(f"WEB trust: {web_trust}")
        
        # Test constraint logic
        web_can_attend_system = web_trust >= system_trust
        user_can_attend_system = user_trust >= system_trust
        system_can_attend_web = system_trust >= web_trust
        
        print(f"\nConstraint tests:")
        print(f"WEB can attend to SYSTEM: {web_can_attend_system} (should be False)")
        print(f"USER can attend to SYSTEM: {user_can_attend_system} (should be False)")
        print(f"SYSTEM can attend to WEB: {system_can_attend_web} (should be True)")
        
        if not web_can_attend_system and not user_can_attend_system and system_can_attend_web:
            print("✅ Trust constraint logic PASSED")
            return True
        else:
            print("❌ Trust constraint logic FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚨 CIV DIAGNOSTIC TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Trust constraint logic", test_simple_trust_constraints),
        ("Basic attention masking", test_basic_attention_masking),
        ("Gradient leak probe", test_gradient_leak_probe),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append(result)
    
    print(f"\n🏆 DIAGNOSTIC RESULTS:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All diagnostics passed - CIV is working correctly!")
    else:
        print("🚨 Some diagnostics failed - CIV needs fixes!")

if __name__ == "__main__":
    main()