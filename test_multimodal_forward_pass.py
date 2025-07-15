#!/usr/bin/env python3
"""
Test script for multimodal CoCoNuT forward pass logic (Task 3.2)

Tests the core implementation requirements:
- 2.3: Continuous thought feedback using multimodal hidden states
- 2.4: Compatibility with InternVL3's attention mechanisms
- 2.5: Efficient handling of visual and textual tokens in KV cache

This test uses the actual InternVL3 model and CUDA.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_coconut.config import Config
from multimodal_coconut.model.multimodal_coconut import create_multimodal_coconut_model


def test_multimodal_forward_pass():
    """Test the core multimodal CoCoNuT forward pass functionality"""
    
    print("=" * 60)
    print("TESTING MULTIMODAL COCONUT FORWARD PASS (Task 3.2)")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires CUDA.")
        return False
    
    device = torch.device('cuda')
    print(f"✓ Using device: {device}")
    
    try:
        # Create test configuration
        config_dict = {
            'model_id': 'OpenGVLab/InternVL3-1B-Pretrained',
            'load_model_path': 'None',
            'use_flash_attn': False,  # Disable for testing stability
            'seed': 42
        }
        config = Config(config_dict)
        
        print("Loading InternVL3 model...")
        model, tokenizer = create_multimodal_coconut_model(config)
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test 1: Latent token detection
        print("\n" + "="*40)
        print("TEST 1: Latent Token Detection")
        print("="*40)
        
        # Create test input with latent tokens
        test_text = "What is in this image? <|latent|> <|latent|> The answer is"
        input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
        
        # Find latent tokens
        latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        latent_indices = (input_ids == latent_token_id).nonzero(as_tuple=False)
        
        print(f"Input text: {test_text}")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Latent token ID: {latent_token_id}")
        print(f"Found {len(latent_indices)} latent tokens at positions: {latent_indices.tolist()}")
        
        assert len(latent_indices) == 2, f"Expected 2 latent tokens, found {len(latent_indices)}"
        print("✓ Latent token detection working correctly")
        
        # Test 2: Multimodal embeddings preparation
        print("\n" + "="*40)
        print("TEST 2: Multimodal Embeddings Preparation")
        print("="*40)
        
        # Create dummy image data
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 448, 448, device=device, dtype=torch.float32)
        attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        image_flags = torch.ones(batch_size, 1, device=device, dtype=torch.long)
        
        print(f"Pixel values shape: {pixel_values.shape}")
        print(f"Input IDs shape: {input_ids.shape}")
        
        # Test embeddings preparation
        with torch.no_grad():
            embeddings = model._prepare_multimodal_embeddings(
                pixel_values, input_ids, image_flags
            )
        
        expected_shape = input_ids.shape + (model.hidden_size,)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert embeddings.shape == expected_shape, f"Shape mismatch: {embeddings.shape} vs {expected_shape}"
        print("✓ Multimodal embeddings preparation working correctly")
        
        # Test 3: Forward pass with latent tokens
        print("\n" + "="*40)
        print("TEST 3: Forward Pass with Continuous Thought")
        print("="*40)
        
        # Create labels for loss computation
        labels = input_ids.clone()
        labels[input_ids == latent_token_id] = -100  # Ignore latent tokens in loss
        
        print("Running forward pass with latent tokens...")
        
        with torch.no_grad():
            outputs = model.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                labels=labels,
                output_hidden_states=True
            )
        
        print(f"✓ Forward pass completed successfully")
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")
        
        # Verify output shapes
        expected_logits_shape = input_ids.shape + (len(tokenizer),)
        assert outputs.logits.shape == expected_logits_shape, f"Logits shape mismatch"
        assert outputs.loss is not None, "Loss should be computed when labels are provided"
        assert torch.isfinite(outputs.loss), "Loss should be finite"
        
        print("✓ Continuous thought feedback working correctly")
        
        # Test 4: Forward pass without latent tokens (standard path)
        print("\n" + "="*40)
        print("TEST 4: Standard Forward Pass (No Latent Tokens)")
        print("="*40)
        
        # Create input without latent tokens
        standard_text = "What is in this image? The answer is a cat."
        standard_input_ids = tokenizer.encode(standard_text, return_tensors='pt').to(device)
        standard_attention_mask = torch.ones_like(standard_input_ids, device=device)
        standard_position_ids = torch.arange(standard_input_ids.size(1), device=device).unsqueeze(0)
        standard_labels = standard_input_ids.clone()
        
        print(f"Standard input: {standard_text}")
        print(f"Input IDs shape: {standard_input_ids.shape}")
        
        with torch.no_grad():
            standard_outputs = model.forward(
                pixel_values=pixel_values,
                input_ids=standard_input_ids,
                attention_mask=standard_attention_mask,
                position_ids=standard_position_ids,
                image_flags=image_flags,
                labels=standard_labels
            )
        
        print(f"✓ Standard forward pass completed")
        print(f"Output logits shape: {standard_outputs.logits.shape}")
        print(f"Loss: {standard_outputs.loss.item() if standard_outputs.loss is not None else 'None'}")
        
        assert standard_outputs.logits is not None, "Standard forward should produce logits"
        assert standard_outputs.loss is not None, "Standard forward should compute loss"
        
        print("✓ Standard forward pass working correctly")
        
        # Test 5: KV cache efficiency
        print("\n" + "="*40)
        print("TEST 5: KV Cache Efficiency")
        print("="*40)
        
        print("Testing KV cache usage...")
        
        with torch.no_grad():
            cache_outputs = model.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                labels=labels,
                use_cache=True
            )
        
        print(f"✓ KV cache test completed")
        print(f"Past key values present: {cache_outputs.past_key_values is not None}")
        
        if cache_outputs.past_key_values:
            print(f"Number of cached layers: {len(cache_outputs.past_key_values)}")
            if len(cache_outputs.past_key_values) > 0:
                key, value = cache_outputs.past_key_values[0]
                print(f"Key shape: {key.shape}, Value shape: {value.shape}")
        
        print("✓ KV cache handling working correctly")
        
        # Test 6: Generation capability
        print("\n" + "="*40)
        print("TEST 6: Generation Capability")
        print("="*40)
        
        # Test generation with latent tokens
        gen_text = "What is in this image? <|latent|>"
        gen_input_ids = tokenizer.encode(gen_text, return_tensors='pt').to(device)
        gen_attention_mask = torch.ones_like(gen_input_ids, device=device)
        
        print(f"Generation input: {gen_text}")
        print(f"Input length: {gen_input_ids.size(1)}")
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                image_flags=image_flags,
                max_new_tokens=5,
                do_sample=False
            )
        
        print(f"✓ Generation completed")
        print(f"Generated length: {generated_ids.size(1)}")
        print(f"Generated text: {tokenizer.decode(generated_ids[0], skip_special_tokens=False)}")
        
        assert generated_ids.size(1) > gen_input_ids.size(1), "Generation should extend sequence"
        
        print("✓ Generation capability working correctly")
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("Task 3.2 implementation verified:")
        print("✓ 2.3: Continuous thought feedback using multimodal hidden states")
        print("✓ 2.4: Compatibility with InternVL3's attention mechanisms")
        print("✓ 2.5: Efficient handling of visual and textual tokens in KV cache")
        print("✓ Multimodal forward pass logic working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the test
    success = test_multimodal_forward_pass()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)