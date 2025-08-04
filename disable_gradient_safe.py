#!/usr/bin/env python3

"""
Temporarily disable gradient-safe masking to test if core CIV works
"""

def disable_gradient_safe():
    """Temporarily disable gradient-safe masking to isolate the core issue"""
    
    # Read the current file
    with open('civ_core.py', 'r') as f:
        content = f.read()
    
    # Find the line that calls gradient-safe masking
    old_line = "            masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)"
    new_line = "            # DISABLED: masked_key_states = self._apply_gradient_safe_mask(key_states, trust_mask)\n            masked_key_states = key_states  # Use unmasked keys for now"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the patched file
        with open('civ_core.py', 'w') as f:
            f.write(content)
        
        print("✅ Temporarily disabled gradient-safe masking")
        print("🔧 Using unmasked keys to test core CIV functionality")
        return True
    else:
        print("❌ Could not find gradient-safe masking call to disable")
        return False

if __name__ == "__main__":
    success = disable_gradient_safe()
    if success:
        print("\n🎯 Test core CIV without gradient-safe masking:")
        print("   python debug_civ_issues.py")
    else:
        print("\n❌ Could not disable - check manually")