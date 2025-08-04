#!/usr/bin/env python3

"""
Quick disable gradient-safe masking to preserve core functionality
"""

def quick_disable():
    """Quickly disable gradient-safe masking"""
    
    with open('civ_core.py', 'r') as f:
        content = f.read()
    
    # Disable the call
    content = content.replace(
        "            masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)",
        "            # TEMP DISABLED: masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)\n            masked_key_states = key_states  # Use unmasked keys"
    )
    
    with open('civ_core.py', 'w') as f:
        f.write(content)
    
    print("✅ Temporarily disabled gradient-safe masking")

if __name__ == "__main__":
    quick_disable()