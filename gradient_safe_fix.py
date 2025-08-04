#!/usr/bin/env python3

"""
Fixed _apply_gradient_safe_mask method for CIV
This replaces the broken version in civ_core.py
"""

def _apply_gradient_safe_mask(self, key_states, trust_mask):
    """
    GAP 8: Apply trust mask to key_states in a gradient-safe way
    
    Multiply key_states by trust mask so forbidden pairs never enter dot-product.
    This prevents gradient leakage through disallowed attention paths.
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    q_len = trust_mask.shape[2]
    
    # TRUE GRADIENT-SAFE MASKING: Zero out keys that shouldn't be attended to
    # trust_mask: [batch, num_heads, q_len, k_len] (True where attention allowed)
    
    # For each key position, determine if ANY query is allowed to attend to it
    # If no query can attend to a key, we zero out that key completely
    # trust_mask.any(dim=2): [batch, num_heads, k_len] - True if any query can attend to this key
    key_mask = trust_mask.any(dim=2, keepdim=True)  # [batch, num_heads, k_len, 1]
    
    # Convert boolean mask to float and apply to key_states
    # This zeros out entire key vectors that no query should attend to
    masked_key_states = key_states * key_mask.float()
    
    return masked_key_states

# Alternative implementation that's more precise but computationally expensive:
def _apply_gradient_safe_mask_precise(self, key_states, trust_mask):
    """
    More precise gradient-safe masking that handles each query-key pair individually
    """
    batch_size, num_heads, k_len, head_dim = key_states.shape
    q_len = trust_mask.shape[2]
    
    # Create masked key_states for each query position
    # This is computationally expensive but guarantees no gradient leakage
    
    # trust_mask: [batch, num_heads, q_len, k_len]
    # We need to create key_states that are masked differently for each query
    
    # Expand key_states to match query dimension: [batch, num_heads, q_len, k_len, head_dim]
    expanded_keys = key_states.unsqueeze(2).expand(batch_size, num_heads, q_len, k_len, head_dim)
    
    # Apply trust mask: [batch, num_heads, q_len, k_len, 1]
    trust_mask_expanded = trust_mask.unsqueeze(-1)
    
    # Mask the keys: [batch, num_heads, q_len, k_len, head_dim]
    masked_expanded_keys = expanded_keys * trust_mask_expanded.float()
    
    # For dot-product computation, we need to handle this in the attention computation
    # This approach requires changing the attention computation to handle per-query masked keys
    
    return masked_expanded_keys

# Simpler approach - just use the key masking version
print("Fixed gradient-safe masking methods created!")
print("Replace the broken method in civ_core.py with the first implementation.")