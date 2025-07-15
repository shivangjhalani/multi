#!/usr/bin/env python3
"""
Test script for multimodal CoCoNuT forward pass implementation.
This tests the core CoCoNuT logic with multimodal inputs.
"""

import torch
import torch.nn as nn
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut

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
    print("\n--- Test 1: Text-only forward pass ---")
    batch_size = 2
    seq_len = 10
    
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
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Text-only forward pass failed: {e}")
        return False
    
    # Test 2: Multimodal forward pass (no latent tokens)
    print("\n--- Test 2: Multimodal forward pass ---")
    
    # Create mock image data
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    image_flags = torch.ones(batch_size, 1)
    
    # Create input with IMG_CONTEXT tokens
    img_context_id = base_model.img_context_token_id
    input_ids_with_img = torch.cat([
        torch.full((batch_size, 5), img_context_id),  # 5 IMG_CONTEXT tokens
        torch.randint(0, 32000, (batch_size, seq_len - 5))
    ], dim=1)
    attention_mask_with_img = torch.ones(batch_size, seq_len)
    labels_with_img = input_ids_with_img.clone()
    
    try:
        outputs = coconut_model(
            pixel_values=pixel_values,
            input_ids=input_ids_with_img,
            attention_mask=attention_mask_with_img,
            image_flags=image_flags,
            labels=labels_with_img
        )
        print(f"✓ Multimodal forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Multimodal forward pass failed: {e}")
        return False
    
    # Test 3: CoCoNuT forward pass with latent tokens
    print("\n--- Test 3: CoCoNuT forward pass with latent tokens ---")
    
    latent_token_id = coconut_model.latent_token_id
    
    # Create input with latent tokens
    input_ids_with_latent = torch.cat([
        torch.randint(0, 32000, (batch_size, 3)),  # Regular tokens
        torch.full((batch_size, 2), latent_token_id),  # 2 latent tokens
        torch.randint(0, 32000, (batch_size, seq_len - 5))  # More regular tokens
    ], dim=1)
    attention_mask_with_latent = torch.ones(batch_size, seq_len)
    labels_with_latent = input_ids_with_latent.clone()
    
    try:
        outputs = coconut_model(
            pixel_values=None,
            input_ids=input_ids_with_latent,
            attention_mask=attention_mask_with_latent,
            labels=labels_with_latent
        )
        print(f"✓ CoCoNuT forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ CoCoNuT forward pass failed: {e}")
        return False
    
    # Test 4: Multimodal CoCoNuT forward pass
    print("\n--- Test 4: Multimodal CoCoNuT forward pass ---")
    
    # Create input with both IMG_CONTEXT and latent tokens
    input_ids_multimodal_coconut = torch.cat([
        torch.full((batch_size, 3), img_context_id),  # IMG_CONTEXT tokens
        torch.randint(0, 32000, (batch_size, 2)),  # Regular tokens
        torch.full((batch_size, 2), latent_token_id),  # Latent tokens
        torch.randint(0, 32000, (batch_size, seq_len - 7))  # More regular tokens
    ], dim=1)
    attention_mask_multimodal = torch.ones(batch_size, seq_len)
    labels_multimodal = input_ids_multimodal_coconut.clone()
    
    try:
        outputs = coconut_model(
            pixel_values=pixel_values,
            input_ids=input_ids_multimodal_coconut,
            attention_mask=attention_mask_multimodal,
            image_flags=image_flags,
            labels=labels_multimodal
        )
        print(f"✓ Multimodal CoCoNuT forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"✗ Multimodal CoCoNuT forward pass failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_multimodal_coconut_forward()
    if success:
        print("\n✓ All multimodal CoCoNuT tests passed!")
    else:
        print("\n✗ Some tests failed!")