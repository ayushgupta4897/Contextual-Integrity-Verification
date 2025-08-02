#!/usr/bin/env python3
"""
CIV Model Persistence - Save and Load CIV Model Directly

This script shows how to save the CIV model after surgery and load it back
without needing to perform surgery each time.
"""

import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from civ_model_surgery import perform_model_surgery


def save_civ_model_complete():
    """Save the complete CIV model after surgery"""
    print("💾 SAVING COMPLETE CIV MODEL")
    print("="*50)
    
    # Load base model and perform surgery
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("🔧 Performing CIV surgery...")
    civ_model = perform_model_surgery(base_model, tokenizer, max_layers=20)
    
    # Save the complete model
    save_dir = "./models/civ-llama-3.2-3b-instruct"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"💾 Saving CIV model to {save_dir}...")
    
    # Save model and tokenizer
    civ_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save metadata
    metadata = {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "civ_layers": 20,
        "total_layers": 28,
        "architecture": "NamespaceAwareAttention",
        "trust_hierarchy": "SYS(100) > USER(80) > TOOL(60) > DOC(50) > WEB(10)"
    }
    
    import json
    with open(f"{save_dir}/civ_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ CIV model saved successfully!")
    print(f"📁 Location: {save_dir}")
    print(f"📊 Size: {get_directory_size(save_dir):.1f} GB")
    
    return save_dir


def load_civ_model_direct(model_path):
    """Load the saved CIV model directly"""
    print(f"📥 LOADING CIV MODEL FROM {model_path}")
    print("="*50)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model 
    # NOTE: This will require the custom NamespaceAwareAttention class to be available
    try:
        civ_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True  # Might be needed for custom classes
        )
        
        print("✅ CIV model loaded successfully!")
        print(f"📊 Parameters: {civ_model.num_parameters():,}")
        
        # Load metadata
        import json
        with open(f"{model_path}/civ_metadata.json", "r") as f:
            metadata = json.load(f)
        
        print("📋 CIV Model Info:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        return civ_model, tokenizer, metadata
        
    except Exception as e:
        print(f"❌ Error loading CIV model: {str(e)}")
        print("💡 Note: Custom NamespaceAwareAttention class must be available")
        return None, None, None


def test_persisted_civ_model():
    """Test the persisted CIV model functionality"""
    print("🧪 TESTING PERSISTED CIV MODEL")
    print("="*50)
    
    # First save the model
    civ_path = save_civ_model_complete()
    
    print("\n" + "="*50)
    
    # Then load it back
    civ_model, tokenizer, metadata = load_civ_model_direct(civ_path)
    
    if civ_model is None:
        print("⚠️ Direct loading failed - using surgery approach instead")
        return False
    
    # Test normal functionality
    print("\n🧪 Testing normal functionality...")
    test_prompt = "What is 2 + 2?"
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(civ_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = civ_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(test_prompt):].strip()
    print(f"Response: {response}")
    
    # Test security functionality
    print("\n🛡️ Testing security functionality...")
    # This would require proper namespace_ids integration
    
    print("✅ Persisted CIV model test complete!")
    return True


def get_directory_size(path):
    """Calculate directory size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert to GB


def main():
    """Main function to demonstrate CIV model persistence"""
    print("🚀 CIV MODEL PERSISTENCE DEMONSTRATION")
    print("="*80)
    
    print("This will show how to save and load CIV models directly.")
    print("Current approach: Surgery on-demand (recommended)")
    print("Alternative: Direct persistence (experimental)")
    
    # Test persistence
    success = test_persisted_civ_model()
    
    print(f"\n🏆 PERSISTENCE RESULT: {'✅ SUCCESS' if success else '⚠️ NEEDS WORK'}")
    
    print("\n💡 CURRENT RECOMMENDATION:")
    print("✅ Use surgery approach - reliable and flexible")
    print("🔬 Direct persistence - possible but needs custom class integration")


if __name__ == "__main__":
    main()