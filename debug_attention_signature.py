#!/usr/bin/env python3
"""
Debug Attention Signature - Check what original LlamaAttention returns
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect


def check_original_attention():
    """Check the original LlamaAttention signature and return values"""
    print("🔍 DEBUGGING ORIGINAL LLAMA ATTENTION")
    print("="*50)
    
    # Load a small model to check attention signature
    print("📥 Loading baseline model...")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Get first attention layer
    first_layer = model.model.layers[0].self_attn
    print(f"Original attention type: {type(first_layer)}")
    
    # Check forward signature
    forward_sig = inspect.signature(first_layer.forward)
    print(f"Forward signature: {forward_sig}")
    
    # Test what it actually returns
    print("\n🧪 Testing actual return values...")
    
    # Create dummy input
    batch_size, seq_len, hidden_size = 1, 5, model.config.hidden_size
    dummy_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16).to(model.device)
    
    with torch.no_grad():
        try:
            result = first_layer(dummy_input)
            print(f"Return type: {type(result)}")
            if isinstance(result, tuple):
                print(f"Tuple length: {len(result)}")
                for i, item in enumerate(result):
                    if item is not None:
                        print(f"  Item {i}: {type(item)} shape {item.shape if hasattr(item, 'shape') else 'N/A'}")
                    else:
                        print(f"  Item {i}: None")
            else:
                print(f"Single return: {type(result)} shape {result.shape if hasattr(result, 'shape') else 'N/A'}")
                
        except Exception as e:
            print(f"❌ Error calling attention: {str(e)}")
    
    print("\n✅ Original attention analysis complete")


def check_usage_in_model():
    """Check how attention is actually used in the model"""
    print("\n🔍 CHECKING ATTENTION USAGE IN MODEL")
    print("="*40)
    
    # Look at the layer forward method
    model = AutoModelForCausalLM.from_pretrained(
        "./models/llama-3.2-3b-instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    first_layer = model.model.layers[0]
    layer_forward_sig = inspect.signature(first_layer.forward)
    print(f"Layer forward signature: {layer_forward_sig}")
    
    # Try to see the source code of how attention is called
    try:
        import transformers
        import inspect
        source = inspect.getsource(first_layer.forward)
        print("Layer forward source (first 500 chars):")
        print(source[:500])
    except Exception as e:
        print(f"Couldn't get source: {e}")


if __name__ == "__main__":
    check_original_attention()
    check_usage_in_model()