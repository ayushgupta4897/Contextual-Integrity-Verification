#!/usr/bin/env python3
"""
Minimal Attention Test - Isolate the exact issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import perform_model_surgery


def test_single_attention_layer():
    """Test a single CIV attention layer in isolation"""
    print("🔍 TESTING SINGLE CIV ATTENTION LAYER")
    print("="*50)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Perform surgery
    civ_model = perform_model_surgery(model, tokenizer)
    
    # Get first CIV attention layer
    first_civ_layer = civ_model.model.layers[0].self_attn
    print(f"CIV attention type: {type(first_civ_layer)}")
    print(f"Has _apply_namespace_mask: {hasattr(first_civ_layer, '_apply_namespace_mask')}")
    
    # Test the attention layer directly
    batch_size, seq_len, hidden_size = 1, 3, model.config.hidden_size
    dummy_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16).to(model.device)
    
    print(f"\n🧪 Testing attention layer directly...")
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            # Call the attention layer directly
            result = first_civ_layer(dummy_input)
            
            print(f"✅ Direct call succeeded!")
            print(f"Result type: {type(result)}")
            
            if isinstance(result, tuple):
                print(f"Tuple length: {len(result)}")
                for i, item in enumerate(result):
                    if item is not None:
                        print(f"  Item {i}: {type(item)}, shape {item.shape}")
                    else:
                        print(f"  Item {i}: None")
            else:
                print(f"Single result: {type(result)}, shape {getattr(result, 'shape', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Direct attention call failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_generation():
    """Test if the issue is in model generation"""
    print("\n🔍 TESTING MODEL GENERATION")
    print("="*40)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Perform surgery
    civ_model = perform_model_surgery(model, tokenizer)
    
    # Simple generation test
    test_prompt = "Hello"
    print(f"Testing prompt: '{test_prompt}'")
    
    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
        
        print("🔄 Attempting generation...")
        with torch.no_grad():
            outputs = civ_model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation succeeded: '{response}'")
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🚀 MINIMAL CIV ATTENTION TEST")
    print("="*60)
    
    # Test 1: Direct attention layer
    attention_ok = test_single_attention_layer()
    
    if attention_ok:
        print("\n✅ Attention layer works - testing generation...")
        # Test 2: Model generation
        generation_ok = test_model_generation()
        
        if generation_ok:
            print("\n🎉 Both attention and generation work!")
        else:
            print("\n⚠️  Attention works but generation fails")
    else:
        print("\n❌ Attention layer is broken")


if __name__ == "__main__":
    main()