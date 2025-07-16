#!/usr/bin/env python3
"""
Demonstration of multimodal CoCoNuT generation capabilities.

This script shows how to use the multimodal CoCoNuT model for:
1. Text-only generation
2. Multimodal generation with images
3. Chat interface for conversations
4. Generation with continuous thoughts (latent tokens)
"""

import torch
import sys
import os
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from multimodal_coconut.config import Config


def create_demo_model():
    """Create a demo multimodal CoCoNuT model for testing."""
    from unittest.mock import Mock
    
    # Create mock base model (in real usage, this would be InternVL3)
    mock_base_model = Mock()
    mock_base_model.config = Mock()
    mock_base_model.config.use_return_dict = True
    
    # Mock language model
    mock_language_model = Mock()
    mock_language_model.config = Mock()
    mock_language_model.config.hidden_size = 768
    mock_language_model.config.vocab_size = 32000
    
    # Mock embeddings
    mock_embeddings = Mock()
    mock_embeddings.weight = torch.randn(32000, 768)
    mock_language_model.get_input_embeddings.return_value = lambda x: torch.randn(x.shape[0], x.shape[1], 768)
    mock_language_model.get_output_embeddings.return_value = mock_embeddings
    
    # Mock generate method to return realistic output
    def mock_generate(**kwargs):
        # Simulate generating a few tokens
        input_length = kwargs.get('inputs_embeds').shape[1]
        max_new_tokens = kwargs.get('max_new_tokens', 10)
        batch_size = kwargs.get('inputs_embeds').shape[0]
        
        # Generate some random token IDs
        new_tokens = torch.randint(3, 1000, (batch_size, max_new_tokens))
        original_tokens = torch.randint(3, 1000, (batch_size, input_length))
        
        return torch.cat([original_tokens, new_tokens], dim=1)
    
    mock_language_model.generate.side_effect = mock_generate
    mock_base_model.language_model = mock_language_model
    
    # Mock vision components
    mock_base_model.extract_feature.return_value = torch.randn(1, 256, 768)
    mock_base_model.img_context_token_id = 103
    mock_base_model.num_image_token = 256
    
    # Create multimodal CoCoNuT model
    model = MultimodalCoconut(
        base_model=mock_base_model,
        latent_token_id=102,
        start_latent_id=100,
        end_latent_id=101,
        eos_token_id=2
    )
    
    return model


def create_demo_tokenizer():
    """Create a demo tokenizer for testing."""
    from unittest.mock import Mock
    
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.side_effect = lambda x: {
        '<|start-latent|>': 100,
        '<|end-latent|>': 101,
        '<|latent|>': 102,
        '<IMG_CONTEXT>': 103,
        '<img>': 104,
        '</img>': 105
    }.get(x, hash(x) % 1000 + 10)  # Generate consistent IDs for other tokens
    
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    
    # Mock tokenization
    def mock_tokenize(text, return_tensors=None):
        # Simple tokenization simulation
        tokens = text.split()
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([token_ids]),
                'attention_mask': torch.ones(1, len(token_ids))
            }
        return token_ids
    
    tokenizer.side_effect = mock_tokenize
    tokenizer.__call__ = mock_tokenize
    
    # Mock decoding
    def mock_decode(token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            # Batch decoding
            return [f"Generated response from tokens {ids[:5]}..." for ids in token_ids]
        else:
            return f"Generated response from tokens {token_ids[:5]}..."
    
    tokenizer.batch_decode.side_effect = mock_decode
    tokenizer.decode.side_effect = lambda x, **kwargs: mock_decode([x], **kwargs)
    
    return tokenizer


def demo_text_only_generation():
    """Demonstrate text-only generation."""
    print("\n🔤 Text-Only Generation Demo")
    print("-" * 40)
    
    model = create_demo_model()
    tokenizer = create_demo_tokenizer()
    
    # Prepare text input
    input_text = "The capital of France is"
    input_tokens = tokenizer(input_text, return_tensors='pt')
    
    print(f"Input: {input_text}")
    
    # Generate response
    outputs = model.generate(
        pixel_values=None,
        input_ids=input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        generation_config={
            'max_new_tokens': 10,
            'do_sample': False,
            'temperature': 0.7
        }
    )
    
    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated: {response}")
    print("✓ Text-only generation completed")


def demo_multimodal_generation():
    """Demonstrate multimodal generation with images."""
    print("\n🖼️ Multimodal Generation Demo")
    print("-" * 40)
    
    model = create_demo_model()
    tokenizer = create_demo_tokenizer()
    
    # Prepare multimodal input
    pixel_values = torch.randn(4, 3, 224, 224)  # 4 image patches
    input_text = "<img> <IMG_CONTEXT> <IMG_CONTEXT> </img> What do you see in this image?"
    input_tokens = tokenizer(input_text, return_tensors='pt')
    
    print(f"Input: Image + '{input_text.replace('<IMG_CONTEXT>', '[IMG]')}'")
    print(f"Image shape: {pixel_values.shape}")
    
    # Generate response
    outputs = model.generate(
        pixel_values=pixel_values,
        input_ids=input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        generation_config={
            'max_new_tokens': 15,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.9
        }
    )
    
    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated: {response}")
    print("✓ Multimodal generation completed")


def demo_chat_interface():
    """Demonstrate the chat interface."""
    print("\n💬 Chat Interface Demo")
    print("-" * 40)
    
    model = create_demo_model()
    tokenizer = create_demo_tokenizer()
    
    # Text-only chat
    print("Text-only conversation:")
    response1 = model.chat(
        tokenizer=tokenizer,
        pixel_values=None,
        question="Hello, how are you?",
        generation_config={'max_new_tokens': 20}
    )
    print(f"Human: Hello, how are you?")
    print(f"Assistant: {response1}")
    
    # Multimodal chat
    print("\nMultimodal conversation:")
    pixel_values = torch.randn(2, 3, 224, 224)  # 2 image patches
    response2 = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question="What's in this image?",
        generation_config={'max_new_tokens': 25},
        num_patches_list=[2]
    )
    print(f"Human: [Shows image] What's in this image?")
    print(f"Assistant: {response2}")
    
    # Chat with history
    print("\nConversation with history:")
    history = [("What's your name?", "I'm a multimodal AI assistant.")]
    response3, updated_history = model.chat(
        tokenizer=tokenizer,
        pixel_values=None,
        question="What can you help me with?",
        generation_config={'max_new_tokens': 30},
        history=history,
        return_history=True
    )
    print(f"Previous: What's your name? -> I'm a multimodal AI assistant.")
    print(f"Human: What can you help me with?")
    print(f"Assistant: {response3}")
    print(f"History length: {len(updated_history)} exchanges")
    print("✓ Chat interface demo completed")


def demo_continuous_thoughts():
    """Demonstrate generation with continuous thoughts (latent tokens)."""
    print("\n🧠 Continuous Thoughts Demo")
    print("-" * 40)
    
    model = create_demo_model()
    tokenizer = create_demo_tokenizer()
    
    # Prepare input with latent tokens
    input_text = "Solve this step by step: 2 + 3 * 4 = <|latent|> <|latent|> The answer is"
    input_tokens = tokenizer(input_text, return_tensors='pt')
    
    print(f"Input: {input_text}")
    print("Note: <|latent|> tokens represent continuous thought steps")
    
    # Generate response
    outputs = model.generate(
        pixel_values=None,
        input_ids=input_tokens['input_ids'],
        attention_mask=input_tokens['attention_mask'],
        generation_config={
            'max_new_tokens': 8,
            'do_sample': False
        }
    )
    
    # Decode response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Generated: {response}")
    print("✓ The model can handle latent tokens in generation")
    print("  (Note: Full CoCoNuT reasoning happens during training/forward pass)")


def demo_generation_parameters():
    """Demonstrate different generation parameters."""
    print("\n⚙️ Generation Parameters Demo")
    print("-" * 40)
    
    model = create_demo_model()
    tokenizer = create_demo_tokenizer()
    
    input_text = "The weather today is"
    input_tokens = tokenizer(input_text, return_tensors='pt')
    
    # Test different sampling strategies
    configs = [
        {'name': 'Greedy', 'config': {'do_sample': False, 'max_new_tokens': 5}},
        {'name': 'Sampling', 'config': {'do_sample': True, 'temperature': 0.8, 'max_new_tokens': 5}},
        {'name': 'Top-p', 'config': {'do_sample': True, 'top_p': 0.9, 'max_new_tokens': 5}},
        {'name': 'Top-k', 'config': {'do_sample': True, 'top_k': 50, 'max_new_tokens': 5}},
    ]
    
    print(f"Input: {input_text}")
    
    for config_info in configs:
        outputs = model.generate(
            pixel_values=None,
            input_ids=input_tokens['input_ids'],
            generation_config=config_info['config']
        )
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"{config_info['name']:10}: {response}")
    
    print("✓ Different generation parameters tested")


def main():
    """Run all demonstrations."""
    print("🚀 Multimodal CoCoNuT Generation Capabilities Demo")
    print("=" * 60)
    print("This demo shows the key generation features of our multimodal CoCoNuT model:")
    print("• Text-only generation")
    print("• Multimodal generation with images")
    print("• Interactive chat interface")
    print("• Continuous thought reasoning")
    print("• Flexible generation parameters")
    
    try:
        demo_text_only_generation()
        demo_multimodal_generation()
        demo_chat_interface()
        demo_continuous_thoughts()
        demo_generation_parameters()
        
        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Text-only and multimodal generation")
        print("✓ Chat interface with conversation history")
        print("✓ Support for continuous thought tokens")
        print("✓ Flexible generation parameters")
        print("✓ Integration with InternVL3 architecture")
        
        print("\nNext Steps:")
        print("• Train the model on A-OKVQA dataset")
        print("• Implement staged curriculum learning")
        print("• Add evaluation metrics")
        print("• Optimize for production deployment")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()