#!/usr/bin/env python3

"""
Patch script to fix the broken _apply_gradient_safe_mask method in civ_core.py
"""

def patch_civ_core():
    """Apply the gradient-safe masking fix to civ_core.py"""
    
    # Read the current file
    with open('civ_core.py', 'r') as f:
        lines = f.readlines()
    
    # Find the lines to replace (around line 475-485)
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if "For true gradient safety, we need to create masked copies" in line:
            start_line = i
        if start_line is not None and "return key_states" in line and i > start_line:
            end_line = i
            break
    
    if start_line is None or end_line is None:
        print("❌ Could not find the method to patch!")
        return False
    
    print(f"📍 Found method to patch at lines {start_line+1}-{end_line+1}")
    
    # Create the replacement lines
    replacement_lines = [
        "        # TRUE GRADIENT-SAFE MASKING: Zero out keys that shouldn't be attended to\n",
        "        # trust_mask: [batch, num_heads, q_len, k_len] (True where attention allowed)\n",
        "        \n",
        "        # For each key position, determine if ANY query is allowed to attend to it\n",
        "        # If no query can attend to a key, we zero out that key completely\n",
        "        key_mask = trust_mask.any(dim=2, keepdim=True)  # [batch, num_heads, k_len, 1]\n",
        "        \n",
        "        # Convert boolean mask to float and apply to key_states\n",
        "        # This zeros out entire key vectors that no query should attend to\n",
        "        masked_key_states = key_states * key_mask.float()\n",
        "        \n",
        "        return masked_key_states\n"
    ]
    
    # Replace the lines
    new_lines = lines[:start_line] + replacement_lines + lines[end_line+1:]
    
    # Write the patched file
    with open('civ_core.py', 'w') as f:
        f.writelines(new_lines)
    
    print("✅ Successfully patched _apply_gradient_safe_mask method!")
    print("🔧 Applied true gradient-safe masking with key masking")
    
    return True

if __name__ == "__main__":
    success = patch_civ_core()
    if success:
        print("\n🎯 Next: Run diagnostic test to verify the fix")
        print("   python debug_civ_issues.py")
    else:
        print("\n❌ Patch failed - manual fix needed")