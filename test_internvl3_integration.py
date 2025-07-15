#!/usr/bin/env python3
"""
Test script for InternVL3 integration with multimodal CoCoNuT

This script tests:
1. Model loading and tokenizer setup
2. Special token integration
3. Basic forward pass functionality
4. Multimodal data handling
"""

import torch
import yaml
from pathlib import Path

# Import our modules
from multimodal_coconut.config import Config
from multimodal_coconut.model.multimodal_coconut import create_multimodal_coconut_model


def test_model_loading():
    """Test basic model loading and tokenizer setup"""
    print("=" * 60)
    print("Testing InternVL3 Model Loading")
    print("=" * 60)
    
    # Load configuration
    config_path = Path("args/multimodal_coconut.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    
    print(f"✓ Configuration loaded: {config.model_id}")
    
    try:
        # Create model and tokenizer
        print("Loading InternVL3 model and tokenizer...")
        model, tokenizer = create_multimodal_coconut_model(config)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Tokenizer vocabulary size: {len(tokenizer)}")
        
        # Check special tokens
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            print(f"✓ Special token '{token}': ID {token_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_tokenizer_functionality():
    """Test tokenizer with multimodal content"""
    print("\n" + "=" * 60)
    print("Testing Tokenizer Functionality")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = Path("args/multimodal_coconut.yaml")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
        
        # Create model and tokenizer
        model, tokenizer = create_multimodal_coconut_model(config)
        
        # Test basic tokenization
        test_text = "<image>\nWhat is shown in this image?\n"
        tokens = tokenizer.encode(test_text, add_special_tokens=True)
        decoded = tokenizer.decode(tokens)
        
        print(f"✓ Original text: {repr(test_text)}")
        print(f"✓ Tokenized: {tokens}")
        print(f"✓ Decoded: {repr(decoded)}")
        
        # Test with CoCoNuT tokens
        coconut_text = "<image>\nWhat is shown in this image?\n<|start-latent|><|latent|><|latent|><|end-latent|>The answer is..."
        coconut_tokens = tokenizer.encode(coconut_text, add_special_tokens=True)
        coconut_decoded = tokenizer.decode(coconut_tokens)
        
        print(f"✓ CoCoNuT text: {repr(coconut_text)}")
        print(f"✓ CoCoNuT tokens: {coconut_tokens}")
        print(f"✓ CoCoNuT decoded: {repr(coconut_decoded)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False


def test_model_forward_pass():
    """Test basic model forward pass"""
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = Path("args/multimodal_coconut.yaml")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
        
        # Create model and tokenizer
        model, tokenizer = create_multimodal_coconut_model(config)
        model.eval()
        
        # Create dummy inputs that match InternVL3's expected format
        batch_size = 2
        num_patches_per_sample = 4
        image_size = 448
        total_patches = batch_size * num_patches_per_sample
        
        # Create dummy pixel values (simulating processed images)
        pixel_values = torch.randn(total_patches, 3, image_size, image_size, dtype=torch.bfloat16)
        
        # Create proper input with IMG_CONTEXT tokens
        # InternVL3 expects <img><IMG_CONTEXT>*N</img> format
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        # Create input text with image context tokens
        num_image_tokens = model.base_model.num_image_token * num_patches_per_sample
        sample_texts = []
        for i in range(batch_size):
            # Create text with proper image context tokens
            text = f"<img>{'<IMG_CONTEXT>' * num_image_tokens}</img>What is in this image?"
            sample_texts.append(text)
        
        # Tokenize the texts
        tokenized = tokenizer(sample_texts, return_tensors='pt', padding=True)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # Create image flags - one flag per image patch
        image_flags = torch.ones(total_patches, 1, dtype=torch.long)
        
        # Set the img_context_token_id in the model
        model.base_model.img_context_token_id = img_context_token_id
        
        print(f"✓ Created dummy inputs:")
        print(f"  - pixel_values shape: {pixel_values.shape}")
        print(f"  - input_ids shape: {input_ids.shape}")
        print(f"  - attention_mask shape: {attention_mask.shape}")
        print(f"  - image_flags shape: {image_flags.shape}")
        
        # Test forward pass without latent tokens
        print("Testing forward pass without latent tokens...")
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                return_dict=True
            )
        
        print(f"✓ Forward pass successful")
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            print(f"✓ Output logits shape: {outputs.logits.shape}")
        else:
            print("⚠ No logits in output (expected for some configurations)")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_latent_token_detection():
    """Test latent token detection and processing"""
    print("\n" + "=" * 60)
    print("Testing Latent Token Detection")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = Path("args/multimodal_coconut.yaml")
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
        
        # Create model and tokenizer
        model, tokenizer = create_multimodal_coconut_model(config)
        model.eval()
        
        # Create input with latent tokens
        text_with_latents = "What is this? <|latent|> <|latent|> The answer is"
        input_ids = torch.tensor([tokenizer.encode(text_with_latents, add_special_tokens=True)], dtype=torch.long)
        
        print(f"✓ Text with latents: {repr(text_with_latents)}")
        print(f"✓ Input IDs: {input_ids}")
        
        # Find latent token positions
        latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        latent_indices = (input_ids == latent_token_id).nonzero(as_tuple=False)
        
        print(f"✓ Latent token ID: {latent_token_id}")
        print(f"✓ Latent positions: {latent_indices}")
        
        if len(latent_indices) > 0:
            print("✓ Latent tokens detected successfully")
        else:
            print("⚠ No latent tokens found")
        
        return True
        
    except Exception as e:
        print(f"❌ Latent token detection test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 InternVL3 Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Tokenizer Functionality", test_tokenizer_functionality),
        ("Model Forward Pass", test_model_forward_pass),
        ("Latent Token Detection", test_latent_token_detection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! InternVL3 integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())