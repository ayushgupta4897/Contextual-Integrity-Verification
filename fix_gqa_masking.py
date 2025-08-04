#!/usr/bin/env python3

"""
Fix gradient-safe masking for Grouped Query Attention (GQA)
"""

def patch_gqa_masking():
    """Fix the _apply_gradient_safe_mask method to handle GQA properly"""
    
    # Read the current file
    with open('civ_core.py', 'r') as f:
        lines = f.readlines()
    
    # Find the gradient-safe mask method
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if "TRUE GRADIENT-SAFE MASKING: Zero out keys that shouldn't be attended to" in line:
            start_line = i
        if start_line is not None and "return masked_key_states" in line:
            end_line = i
            break
    
    if start_line is None or end_line is None:
        print("❌ Could not find the gradient-safe mask method!")
        return False
    
    print(f"📍 Found method to patch at lines {start_line+1}-{end_line+1}")
    
    # Create the GQA-aware replacement
    replacement_lines = [
        "        # TRUE GRADIENT-SAFE MASKING: Handle Grouped Query Attention (GQA)\n",
        "        # trust_mask: [batch, num_query_heads, q_len, k_len]\n",
        "        # key_states: [batch, num_kv_heads, k_len, head_dim]\n",
        "        \n",
        "        batch_size, num_query_heads, q_len, k_len = trust_mask.shape\n",
        "        batch_size_k, num_kv_heads, k_len_k, head_dim = key_states.shape\n",
        "        \n",
        "        # For GQA: num_query_heads >= num_kv_heads (e.g., 24 query, 8 kv)\n",
        "        # We need to create a key mask that matches the kv dimensions\n",
        "        \n",
        "        if num_query_heads != num_kv_heads:\n",
        "            # Grouped Query Attention: Reduce trust_mask to match kv heads\n",
        "            num_groups = num_query_heads // num_kv_heads\n",
        "            \n",
        "            # Reshape trust_mask to group query heads: [batch, num_kv_heads, num_groups, q_len, k_len]\n",
        "            trust_mask_grouped = trust_mask.view(batch_size, num_kv_heads, num_groups, q_len, k_len)\n",
        "            \n",
        "            # Any query in the group can determine if a key should be accessible\n",
        "            # Take logical OR across the group dimension: [batch, num_kv_heads, q_len, k_len]\n",
        "            trust_mask_kv = trust_mask_grouped.any(dim=2)\n",
        "        else:\n",
        "            # Standard attention: query heads == kv heads\n",
        "            trust_mask_kv = trust_mask\n",
        "        \n",
        "        # For each key position, check if ANY query can attend to it\n",
        "        # key_mask: [batch, num_kv_heads, k_len, 1]\n",
        "        key_mask = trust_mask_kv.any(dim=2, keepdim=True)\n",
        "        \n",
        "        # Apply mask to key_states: zero out keys that no query should attend to\n",
        "        masked_key_states = key_states * key_mask.float()\n",
        "\n",
        "        return masked_key_states\n"
    ]
    
    # Replace the lines
    new_lines = lines[:start_line] + replacement_lines + lines[end_line+1:]
    
    # Write the patched file
    with open('civ_core.py', 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Successfully patched gradient-safe masking for GQA!")
    print("🔧 Now handles Grouped Query Attention properly")
    
    return True

if __name__ == "__main__":
    success = patch_gqa_masking()
    if success:
        print("\n🎯 Next: Test the GQA-aware fix")
        print("   python debug_tensor_shapes.py")
    else:
        print("\n❌ Patch failed - manual fix needed")