#!/usr/bin/env python3

"""
Re-enable the GQA-aware gradient-safe masking call
"""

def reenable_gradient_safe():
    """Re-enable the gradient-safe masking call"""
    
    # Read the current file
    with open('civ_core.py', 'r') as f:
        content = f.read()
    
    # Find and replace the disabled call
    old_text = "            # DISABLED: masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)\n            masked_key_states = key_states  # Use unmasked keys for now"
    
    new_text = "            # GAP 8: Apply GQA-aware gradient-safe masking\n            masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)"
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        
        # Write back to file
        with open('civ_core.py', 'w') as f:
            f.write(content)
        
        print("✅ Re-enabled GQA-aware gradient-safe masking")
        return True
    else:
        print("❌ Could not find disabled gradient-safe masking call")
        return False

if __name__ == "__main__":
    success = reenable_gradient_safe()
    if success:
        print("🎯 Gradient-safe masking is now active with GQA support")
    else:
        print("❌ Manual re-enabling required")