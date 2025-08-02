#!/usr/bin/env python3
"""
Progressive Surgery Test - Find the Breaking Point

Test replacing 1, 2, 5, 10, 20, and all 28 layers to find 
where the model starts breaking down.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import NamespaceAwareAttention


def test_n_layer_replacement(num_layers_to_replace: int):
    """Test replacing the first N attention layers"""
    print(f"\n🧪 TESTING {num_layers_to_replace} LAYER REPLACEMENT")
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
    
    print(f"🔧 Replacing first {num_layers_to_replace} attention layers...")
    
    # Replace the first N layers
    replaced_count = 0
    for i in range(min(num_layers_to_replace, 28)):  # Model has 28 layers
        original_attention = model.model.layers[i].self_attn
        civ_attention = NamespaceAwareAttention(model.config, layer_idx=i)
        
        # Copy weights
        civ_attention.q_proj.load_state_dict(original_attention.q_proj.state_dict())
        civ_attention.k_proj.load_state_dict(original_attention.k_proj.state_dict())
        civ_attention.v_proj.load_state_dict(original_attention.v_proj.state_dict())
        civ_attention.o_proj.load_state_dict(original_attention.o_proj.state_dict())
        
        # Move to device and replace
        civ_attention = civ_attention.to(model.device)
        model.model.layers[i].self_attn = civ_attention
        replaced_count += 1
    
    print(f"✅ Replaced {replaced_count} attention layers")
    
    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?", 
        "What color is the sky?",
    ]
    
    results = []
    for prompt in test_prompts:
        print(f"\n📝 Testing: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            print(f"Response: {response[:60]}...")
            
            # Quality check
            has_repetitive = "QuestionQuestion" in response[:50]
            has_garbled = len(set(response[:20])) < 3  # Very low diversity
            is_reasonable_length = len(response.strip()) > 3
            
            is_good = not has_repetitive and not has_garbled and is_reasonable_length
            status = "✅ GOOD" if is_good else "❌ POOR"
            
            print(f"Quality: {status}")
            results.append(is_good)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append(False)
    
    # Summary
    good_count = sum(results)
    success_rate = good_count / len(test_prompts) * 100
    
    print(f"\n📊 SUMMARY FOR {num_layers_to_replace} LAYERS:")
    print(f"   Good responses: {good_count}/{len(test_prompts)} ({success_rate:.1f}%)")
    
    if success_rate >= 66:
        print(f"   ✅ WORKING: Model functions well with {num_layers_to_replace} layers replaced")
        return True
    else:
        print(f"   ❌ BROKEN: Model degraded with {num_layers_to_replace} layers replaced")
        return False


def main():
    """Test progressive layer replacement"""
    print("🚀 PROGRESSIVE SURGERY TEST - FIND THE BREAKING POINT")
    print("="*70)
    
    # Test different numbers of layers
    layer_counts = [1, 2, 5, 10, 15, 20, 28]
    results = {}
    
    for count in layer_counts:
        print(f"\n{'='*60}")
        success = test_n_layer_replacement(count)
        results[count] = success
        
        if not success:
            print(f"🚨 BREAKING POINT FOUND: Model breaks after {count} layers")
            break
    
    # Analysis
    print(f"\n🏆 PROGRESSIVE SURGERY RESULTS")
    print("="*40)
    for count, success in results.items():
        status = "✅ WORKS" if success else "❌ BREAKS"
        print(f"   {count:2d} layers: {status}")
    
    # Find the exact breaking point
    working_counts = [c for c, s in results.items() if s]
    broken_counts = [c for c, s in results.items() if not s]
    
    if working_counts and broken_counts:
        max_working = max(working_counts)
        min_broken = min(broken_counts)
        print(f"\n🎯 CONCLUSION:")
        print(f"   ✅ Works up to: {max_working} layers")
        print(f"   ❌ Breaks at: {min_broken} layers")
        print(f"   💡 Breaking point is between {max_working} and {min_broken} layers")
    elif not broken_counts:
        print(f"\n🎉 EXCELLENT: All tested configurations work!")
    else:
        print(f"\n⚠️  Model breaks immediately at 1 layer")


if __name__ == "__main__":
    main()