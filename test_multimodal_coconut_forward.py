#!/usr/bin/env python3
"""
Test script for multimodal CoCoNuT forward pass implementation.
This tests the core CoCoNuT logic with multimodal inputs.
"""

import torch
import torch.nn as nn
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from multimodal_coconut.config import Config

def create_mock_internvl_model():
    """Create a mock InternVL3 model for testing"""
    class MockLanguageModel(nn.Module):
        def __init__(self, vocab_size=32000, hidden_size=2048):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=16,
                    batch_first=True
                ),
                num_layers=2
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size)
            self.config = type('Config', (), {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size
            })()
        
        def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, 
                   past_key_values=None, output_hidden_states=False, use_cache=False, **kwargs):
            # Simple forward pass for testing
            batch_size, seq_len, hidden_size = inputs_embeds.shape
            
            # Create dummy hidden states
            hidden_states = inputs_embeds + 0.1 * torch.randn_like(inputs_embeds)
            
            # Create logits
            logits = self.lm_head(hidden_states)
            
            # Create dummy past_key_values if use_cache
            past_key_values_out = None
            if use_cache:
                # Create dummy KV cache
                past_key_values_out = [
                    (torch.randn(batch_size, 16, seq_len, 64), 
                     torch.randn(batch_size, 16, seq_len, 64))
                    for _ in range(2)  # 2 layers
                ]
            
            # Create output object
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values_out,
                hidden_states=[hidden_states] if output_hidden_states else None,
                attentions=None
            )
        
        def get_input_embeddings(self):
            return self.embeddings
        
        def resize_token_embeddings(self, new_size):
            old_embeddings = self.embeddings
            new_embeddings = nn.Embedding(new_size, old_embeddings.embedding_dim)
            # Copy old weights
            new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
            self.embeddings = new_embeddings
            self.lm_head = nn.Linear(old_embeddings.embedding_dim, new_size)
    
    class MockInternVLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = MockLanguageModel()
            self.vision_model = nn.Identity()  # Dummy vision model
            self.img_context_token_id = 32000  # Mock IMG_CONTEXT token ID
            self.num_image_token = 256  # Mock number of image tokens
            
        def extract_feature(self, pixel_values):
            """Mock feature extraction"""
            if pixel_values is None:
                return None
            batch_size = pixel_values.shape[0]
            return torch.randn(batch_size, self.num_image_token, 2048)  # Mock visual features
        
        def forward(self, **kwargs):
            return self.language_model(**kwargs)
    
    return MockInternVLModel()

def test_multimodal_coconut_forward():
    """Test the multimodal CoCoNuT forward pass"""
    print("=== Testing Multimodal CoCoNuT Forward Pass ===")
    
    # Create mock model
    base_model = create_mock_internvl_model()
    
    # Create CoCoNuT wrapper
    coconut_model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=32001,  # <|latent|>
        start_latent_id=32002,  # <|start-latent|>
        end_latent_id=32003,    # <|end-latent|>
        eos_token_id=2          # </s>
    )
    
    print("✓ MultimodalCoconut model created successfully")
    
    # Test 1: Text-only forward pass (no latent tokens)
    print("\n1. Testing text-only forward pass (no latent tokens)...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    try:
        outputs = coconut_model(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        print(f"✓ Text-only forward pass successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Text-only forward pass failed: {e}")
        return False
    
    # Test 2: Multimodal forward pass (no latent tokens)
    print("\n2. Testing multimodal forward pass (no latent tokens)...")
    pixel_values = torch.randn(batch_size, 3, 224, 224)  # Mock images
    image_flags = torch.ones(batch_size, 1)
    
    # Create input with IMG_CONTEXT tokens
    input_ids_with_img = input_ids.clone()
    input_ids_with_img[:, 2:4] = base_model.img_context_token_id  # Replace some tokens with IMG_CONTEXT
    
    try:
        outputs = coconut_model(
            pixel_values=pixel_values,
            input_ids=input_ids_with_img,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=input_ids_with_img
        )
        print(f"✓ Multimodal forward pass successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Multimodal forward pass failed: {e}")
        return False
    
    # Test 3: CoCoNuT forward pass with latent tokens
    print("\n3. Testing CoCoNuT forward pass with latent tokens...")
    
    # Create input with latent tokens
    input_ids_with_latent = input_ids.clone()
    input_ids_with_latent[:, 5:7] = coconut_model.latent_token_id  # Add latent tokens
    
    try:
        outputs = coconut_model(
            pixel_values=None,
            input_ids=input_ids_with_latent,
            attention_mask=attention_mask,
            labels=input_ids_with_latent
        )
        print(f"✓ CoCoNuT forward pass successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ CoCoNuT forward pass failed: {e}")
        return False
    
    # Test 4: Multimodal CoCoNuT forward pass (the full test)
    print("\n4. Testing multimodal CoCoNuT forward pass...")
    
    # Create input with both IMG_CONTEXT and latent tokens
    input_ids_full = input_ids.clone()
    input_ids_full[:, 2:4] = base_model.img_context_token_id  # IMG_CONTEXT tokens
    input_ids_full[:, 6:8] = coconut_model.latent_token_id    # Latent tokens
    
    try:
        outputs = coconut_model(
            pixel_values=pixel_values,
            input_ids=input_ids_full,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=input_ids_full
        )
        print(f"✓ Multimodal CoCoNuT forward pass successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
        print(f"  - Hidden states available: {outputs.hidden_states is not None}")
        print(f"  - Past key values available: {outputs.past_key_values is not None}")
    except Exception as e:
        print(f"✗ Multimodal CoCoNuT forward pass failed: {e}")
        return False
    
    # Test 5: Edge case - multiple latent tokens
    print("\n5. Testing multiple latent tokens...")
    
    input_ids_multi_latent = input_ids.clone()
    input_ids_multi_latent[:, 3:6] = coconut_model.latent_token_id  # 3 consecutive latent tokens
    
    try:
        outputs = coconut_model(
            pixel_values=None,
            input_ids=input_ids_multi_latent,
            attention_mask=attention_mask,
            labels=input_ids_multi_latent
        )
        print(f"✓ Multiple latent tokens test successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Multiple latent tokens test failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    return True

def test_continuous_thought_mechanism():
    """Test that the continuous thought mechanism is working correctly"""
    print("\n=== Testing Continuous Thought Mechanism ===")
    
    # Create a simple test to verify that latent tokens get replaced with hidden states
    base_model = create_mock_internvl_model()
    coconut_model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=32001,
        start_latent_id=32002,
        end_latent_id=32003,
        eos_token_id=2
    )
    
    # Create input with one latent token
    batch_size, seq_len = 1, 5
    input_ids = torch.tensor([[100, 200, 32001, 300, 400]])  # One latent token at position 2
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test that the model processes the latent token correctly
    try:
        outputs = coconut_model(
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        print("✓ Continuous thought mechanism test passed")
        print(f"  - Successfully processed latent token at position 2")
        print(f"  - Output shape: {outputs.logits.shape}")
        
        # The key test: verify that we get the expected sequence length
        expected_seq_len = seq_len
        actual_seq_len = outputs.logits.shape[1]
        
        if actual_seq_len == expected_seq_len:
            print(f"✓ Output sequence length correct: {actual_seq_len}")
        else:
            print(f"✗ Output sequence length mismatch: expected {expected_seq_len}, got {actual_seq_len}")
            return False
            
    except Exception as e:
        print(f"✗ Continuous thought mechanism test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_multimodal_coconut_forward()
    if success:
        success = test_continuous_thought_mechanism()
    
    if success:
        print("\n🎉 All multimodal CoCoNuT tests passed!")
        print("The implementation appears to be working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")