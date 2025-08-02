#!/usr/bin/env python3
"""
Minimal Attention Debug - Find the Core Issue

This creates the most basic test possible to isolate what's wrong 
with our NamespaceAwareAttention implementation.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import NamespaceAwareAttention


def debug_single_attention_layer():
    """Debug just one attention layer in isolation"""
    print("🔍 MINIMAL ATTENTION DEBUG")
    print("="*40)
    
    # Load a model to get config
    print("📥 Loading model config...")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    config = model.config
    
    print("🧪 Testing single NamespaceAwareAttention layer...")
    
    # Create our custom attention
    civ_attention = NamespaceAwareAttention(config, layer_idx=0)
    
    # Get original attention for comparison
    original_attention = model.model.layers[0].self_attn
    
    # Copy weights
    civ_attention.q_proj.load_state_dict(original_attention.q_proj.state_dict())
    civ_attention.k_proj.load_state_dict(original_attention.k_proj.state_dict())
    civ_attention.v_proj.load_state_dict(original_attention.v_proj.state_dict())
    civ_attention.o_proj.load_state_dict(original_attention.o_proj.state_dict())
    
    # Move to same device
    civ_attention = civ_attention.to(model.device)
    
    print("⚡ Running forward pass comparison...")
    
    # Create test input
    batch_size, seq_len, hidden_size = 1, 5, config.hidden_size
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=model.device)
    
    # Test both attentions
    with torch.no_grad():
        try:
            # Original attention
            original_output, original_weights = original_attention(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=True,
                use_cache=False
            )
            print(f"✅ Original attention: {original_output.shape}")
            print(f"   Output sample: {original_output[0, 0, :5]}")
            
            # Our CIV attention (no namespace_ids = normal mode)
            civ_output, civ_weights = civ_attention(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=True,
                use_cache=False,
                namespace_ids=None  # Normal mode
            )
            print(f"✅ CIV attention: {civ_output.shape}")
            print(f"   Output sample: {civ_output[0, 0, :5]}")
            
            # Compare outputs
            diff = torch.abs(original_output - civ_output).max().item()
            print(f"📊 Max difference: {diff}")
            
            if diff < 1e-3:
                print("🎉 EXCELLENT: Outputs are nearly identical!")
            elif diff < 1e-1:
                print("✅ GOOD: Outputs are reasonably close")
            else:
                print("❌ POOR: Outputs are very different")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()


def test_generation_with_minimal_surgery():
    """Test generation with just one layer replaced"""
    print(f"\n🧪 TESTING GENERATION WITH ONE LAYER REPLACEMENT")
    print("="*50)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Replace ONLY the first attention layer
    print("🔧 Replacing only layer 0...")
    original_attention = model.model.layers[0].self_attn
    civ_attention = NamespaceAwareAttention(model.config, layer_idx=0)
    
    # Copy weights
    civ_attention.q_proj.load_state_dict(original_attention.q_proj.state_dict())
    civ_attention.k_proj.load_state_dict(original_attention.k_proj.state_dict())
    civ_attention.v_proj.load_state_dict(original_attention.v_proj.state_dict())
    civ_attention.o_proj.load_state_dict(original_attention.o_proj.state_dict())
    
    # Move to device and replace
    civ_attention = civ_attention.to(model.device)
    model.model.layers[0].self_attn = civ_attention
    
    print("✅ Replaced 1 attention layer")
    
    # Test generation
    prompt = "What is 2 + 2?"
    print(f"📝 Testing: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        print(f"Response: {response}")
        
        has_repetitive = "QuestionQuestion" in response
        print(f"Repetitive pattern: {'❌ YES' if has_repetitive else '✅ NO'}")
        
        if not has_repetitive and len(response.strip()) > 0:
            print("🎉 SUCCESS: Single layer replacement works!")
            return True
        else:
            print("❌ FAILURE: Still has issues with single layer")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Run minimal debugging"""
    print("🚀 MINIMAL CIV ATTENTION DEBUGGING")
    print("="*60)
    
    # Test 1: Single attention layer in isolation
    debug_single_attention_layer()
    
    # Test 2: Generation with minimal surgery
    success = test_generation_with_minimal_surgery()
    
    print(f"\n🏁 DEBUGGING COMPLETE!")
    if success:
        print("✅ Single layer works - issue might be with full model surgery")
    else:
        print("❌ Issue is fundamental - attention layer itself is broken")


if __name__ == "__main__":
    main()