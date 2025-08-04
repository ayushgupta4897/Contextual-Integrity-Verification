#!/usr/bin/env python3

"""
Debug tensor shapes in CIV gradient-safe masking
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_core import CIVProtectedModel

def debug_tensor_shapes():
    """Debug the tensor shape issues in gradient-safe masking"""
    print("🔍 DEBUGGING TENSOR SHAPES")
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
        
        # Get model dimensions
        config = base_model.config
        print(f"Model config:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_attention_heads: {config.num_attention_heads}")
        print(f"  num_key_value_heads: {config.num_key_value_heads}")
        
        head_dim = config.hidden_size // config.num_attention_heads
        print(f"  head_dim: {head_dim}")
        
        protected_model = CIVProtectedModel(base_model, tokenizer, max_layers=1)
        
        # Create test tensors with correct dimensions
        batch_size = 1
        seq_len = 3
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        
        print(f"\nTest tensor dimensions:")
        print(f"  batch_size: {batch_size}")
        print(f"  seq_len: {seq_len}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_heads: {num_heads}")
        print(f"  num_kv_heads: {num_kv_heads}")
        print(f"  head_dim: {head_dim}")
        
        # Create test inputs
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        namespace_ids = torch.tensor([[20, 100, 20]])  # WEB, SYSTEM, WEB
        
        print(f"\nInput tensor shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  namespace_ids: {namespace_ids.shape}")
        
        # Get the CIV attention layer
        civ_attn = protected_model.base_model.model.layers[0].self_attn
        
        # Manually call the attention components to debug shapes
        print(f"\n🔧 DEBUGGING ATTENTION COMPUTATION:")
        
        # QKV projections
        query_states = civ_attn.q_proj(hidden_states)
        key_states = civ_attn.k_proj(hidden_states)
        value_states = civ_attn.v_proj(hidden_states)
        
        print(f"  After projections:")
        print(f"    query_states: {query_states.shape}")
        print(f"    key_states: {key_states.shape}")
        print(f"    value_states: {value_states.shape}")
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        print(f"  After reshape:")
        print(f"    query_states: {query_states.shape}")
        print(f"    key_states: {key_states.shape}")
        print(f"    value_states: {value_states.shape}")
        
        # Create trust mask
        q_len, k_len = seq_len, seq_len
        query_trust = namespace_ids[:, :q_len].unsqueeze(2)  # [batch, q_len, 1]
        key_trust = namespace_ids[:, :k_len].unsqueeze(1)    # [batch, 1, k_len]
        trust_mask = query_trust >= key_trust  # [batch, q_len, k_len]
        
        print(f"  Trust mask creation:")
        print(f"    query_trust: {query_trust.shape}")
        print(f"    key_trust: {key_trust.shape}")
        print(f"    trust_mask: {trust_mask.shape}")
        
        # Expand trust mask for attention heads
        trust_mask = trust_mask.unsqueeze(1).expand(batch_size, num_heads, q_len, k_len)
        print(f"    trust_mask expanded: {trust_mask.shape}")
        
        # Test gradient-safe masking
        print(f"\n🎯 TESTING GRADIENT-SAFE MASKING:")
        
        # Our current approach: key masking
        key_mask = trust_mask.any(dim=2, keepdim=True)  # [batch, num_heads, k_len, 1]
        print(f"    key_mask: {key_mask.shape}")
        
        # Check if we can multiply
        print(f"    key_states shape: {key_states.shape}")
        print(f"    key_mask shape: {key_mask.shape}")
        
        # The issue might be num_heads vs num_kv_heads mismatch
        if num_heads != num_kv_heads:
            print(f"    ⚠️ MISMATCH: num_heads ({num_heads}) != num_kv_heads ({num_kv_heads})")
            print(f"    Need to handle Grouped Query Attention (GQA)")
            
            # For GQA, we need to repeat the key mask to match query heads
            num_key_value_groups = num_heads // num_kv_heads
            print(f"    num_key_value_groups: {num_key_value_groups}")
            
            # Adjust key_mask to match key_states dimensions
            key_mask_adjusted = key_mask.expand(batch_size, num_kv_heads, k_len, 1)
            print(f"    key_mask_adjusted: {key_mask_adjusted.shape}")
            
            try:
                masked_key_states = key_states * key_mask_adjusted.float()
                print(f"    ✅ SUCCESS: masked_key_states: {masked_key_states.shape}")
            except Exception as e:
                print(f"    ❌ FAILED: {e}")
        else:
            try:
                masked_key_states = key_states * key_mask.float()
                print(f"    ✅ SUCCESS: masked_key_states: {masked_key_states.shape}")
            except Exception as e:
                print(f"    ❌ FAILED: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_tensor_shapes()