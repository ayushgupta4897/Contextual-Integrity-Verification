#!/usr/bin/env python3

"""
Implement proper GQA-aware gradient-safe masking
Handles Llama's 24 query heads → 8 key/value heads architecture
"""

def implement_gqa_gradient_safe():
    """Replace the gradient-safe masking method with proper GQA support"""
    
    # Read the current file
    with open('civ_core.py', 'r') as f:
        content = f.read()
    
    # Find the current gradient-safe masking method
    start_marker = "def _apply_gradient_safe_mask(self, key_states, trust_mask):"
    end_marker = "        return masked_key_states"
    
    start_pos = content.find(start_marker)
    if start_pos == -1:
        print("❌ Could not find _apply_gradient_safe_mask method")
        return False
    
    # Find the end of the method
    end_pos = content.find(end_marker, start_pos)
    if end_pos == -1:
        print("❌ Could not find end of _apply_gradient_safe_mask method")
        return False
    
    end_pos += len(end_marker)
    
    # Create the new GQA-aware implementation
    new_method = '''def _apply_gradient_safe_mask(self, key_states, trust_mask):
        """
        GAP 8: GQA-aware gradient-safe masking
        
        Handles Grouped Query Attention where num_query_heads != num_kv_heads
        (e.g., Llama: 24 query heads, 8 key/value heads)
        
        Args:
            key_states: [batch, num_kv_heads, k_len, head_dim]
            trust_mask: [batch, num_query_heads, q_len, k_len]
        """
        batch_size, num_kv_heads, k_len, head_dim = key_states.shape
        batch_size_mask, num_query_heads, q_len, k_len_mask = trust_mask.shape
        
        # Verify dimensions match where expected
        if batch_size != batch_size_mask or k_len != k_len_mask:
            raise ValueError(f"Dimension mismatch: key_states batch/k_len {batch_size}/{k_len} "
                           f"vs trust_mask {batch_size_mask}/{k_len_mask}")
        
        if num_query_heads == num_kv_heads:
            # Standard attention: direct masking
            # For each key, check if ANY query can attend to it
            key_mask = trust_mask.any(dim=2, keepdim=True)  # [batch, num_kv_heads, k_len, 1]
            
        else:
            # GQA: num_query_heads > num_kv_heads (e.g., 24 queries, 8 kv)
            if num_query_heads % num_kv_heads != 0:
                raise ValueError(f"GQA: num_query_heads ({num_query_heads}) must be divisible by "
                               f"num_kv_heads ({num_kv_heads})")
            
            num_groups = num_query_heads // num_kv_heads  # e.g., 24 // 8 = 3
            
            # Reshape trust_mask to group queries: [batch, num_kv_heads, num_groups, q_len, k_len]
            trust_mask_grouped = trust_mask.view(batch_size, num_kv_heads, num_groups, q_len, k_len)
            
            # For each kv head, check if ANY query in ANY group can attend to each key
            # First, any query within each group: [batch, num_kv_heads, num_groups, k_len]
            any_in_group = trust_mask_grouped.any(dim=3)
            
            # Then, any group can determine key accessibility: [batch, num_kv_heads, k_len]
            any_group = any_in_group.any(dim=2)
            
            # Add dimension for broadcasting: [batch, num_kv_heads, k_len, 1]
            key_mask = any_group.unsqueeze(-1)
        
        # Apply gradient-safe masking: zero out keys that no query should attend to
        # This prevents forbidden keys from participating in dot-product computation
        masked_key_states = key_states * key_mask.float()
        
        return masked_key_states'''
    
    # Replace the method
    new_content = content[:start_pos] + new_method + content[end_pos:]
    
    # Write back to file
    with open('civ_core.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Successfully implemented GQA-aware gradient-safe masking!")
    print("🔧 Handles 24 query → 8 kv head architecture properly")
    print("🎯 Key features:")
    print("   - Proper GQA dimension handling")
    print("   - Group-aware trust mask processing")
    print("   - Gradient-safe key masking")
    
    return True

def test_implementation():
    """Test the new implementation with our safety validator"""
    print("\n🧪 Testing GQA-aware implementation...")
    
    import subprocess
    result = subprocess.run(
        ["python", "safety_validator.py"], 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Safety validation PASSED - core functionality intact")
        return True
    else:
        print("❌ Safety validation FAILED")
        print(result.stdout)
        return False

if __name__ == "__main__":
    print("🚀 IMPLEMENTING GQA-AWARE GRADIENT-SAFE MASKING")
    print("=" * 60)
    
    # Step 1: Implement the fix
    success = implement_gqa_gradient_safe()
    if not success:
        print("❌ Implementation failed")
        exit(1)
    
    # Step 2: Test with safety validator
    test_success = test_implementation()
    if test_success:
        print("\n🎉 GQA-aware gradient-safe masking successfully implemented!")
        print("🛡️ Core CIV functionality preserved")
    else:
        print("\n❌ Implementation broke core functionality")
        exit(1)