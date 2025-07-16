#!/usr/bin/env python3
"""
Test the stage integration and visual embedding fix
"""

import sys
import os
sys.path.append('.')

import torch
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModel

from multimodal_coconut.config import Config
from multimodal_coconut.data.dataset import get_multimodal_dataset, MultimodalCollator
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut


def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    model_id = "OpenGVLab/InternVL3-1B-Pretrained"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    tokenizer.add_tokens(special_tokens)
    
    tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    tokenizer.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    tokenizer.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    base_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, device_map="cpu"
    )
    
    if len(tokenizer) > base_model.language_model.config.vocab_size:
        base_model.language_model.resize_token_embeddings(len(tokenizer))
    
    # Set IMG_CONTEXT token ID
    try:
        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        base_model.img_context_token_id = img_context_token_id
        print(f"✓ IMG_CONTEXT token ID: {img_context_token_id}")
    except:
        base_model.img_context_token_id = None
        print("⚠️  IMG_CONTEXT token not found")
    
    return base_model, tokenizer


def create_test_data(temp_dir):
    """Create minimal test data"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create a single test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = images_dir / "test_image.jpg"
    img.save(img_path)
    
    # Create minimal training data
    train_data = [{
        "image_path": str(img_path),
        "question": "What is in this image?",
        "steps": ["Step 1: I can see the image"],
        "answer": "This is a test image"
    }]
    
    train_path = Path(temp_dir) / "train.json"
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    return str(train_path), str(images_dir)


def debug_visual_embedding_mismatch():
    """Debug the visual embedding shape mismatch"""
    print("\n🔍 DEBUGGING VISUAL EMBEDDING SHAPE MISMATCH")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Setup
        base_model, tokenizer = setup_model_and_tokenizer()
        train_path, images_dir = create_test_data(temp_dir)
        
        # Create dataset
        dataset = get_multimodal_dataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_num_patches=4,
            max_size=1
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Examine the sample
        sample = dataset[0]
        print(f"\n📊 Sample analysis:")
        print(f"  Pixel values shape: {sample['pixel_values'].shape}")
        print(f"  Num patches: {sample['num_patches']}")
        
        # Check tokenized question
        question_tokens = sample['question_tokenized']
        print(f"  Question tokens length: {len(question_tokens)}")
        
        # Count IMG_CONTEXT tokens
        img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        img_context_count = question_tokens.count(img_context_id)
        print(f"  IMG_CONTEXT tokens in question: {img_context_count}")
        
        # Create collator and batch
        collator = MultimodalCollator(tokenizer=tokenizer, latent_id=tokenizer.latent_token_id)
        batch = collator([sample])
        
        print(f"\n📦 Batch analysis:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  Image flags shape: {batch['image_flags'].shape}")
        
        # Count IMG_CONTEXT tokens in batch
        input_ids_flat = batch['input_ids'].flatten()
        img_context_count_batch = (input_ids_flat == img_context_id).sum().item()
        print(f"  IMG_CONTEXT tokens in batch: {img_context_count_batch}")
        
        # Create model and test forward pass
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Test forward pass with detailed debugging
        forward_batch = {k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']}
        
        print(f"\n🚀 Testing forward pass...")
        
        try:
            # Let's examine what happens in the base model
            print(f"  Base model img_context_token_id: {base_model.img_context_token_id}")
            
            # Extract visual features manually to see their shape
            pixel_values = forward_batch['pixel_values']
            print(f"  Input pixel values shape: {pixel_values.shape}")
            
            vit_embeds = base_model.extract_feature(pixel_values)
            print(f"  Extracted visual embeddings shape: {vit_embeds.shape}")
            
            # Check how many IMG_CONTEXT tokens we expect vs have
            input_ids = forward_batch['input_ids']
            selected = (input_ids.flatten() == img_context_id)
            selected_count = selected.sum().item()
            print(f"  Selected IMG_CONTEXT positions: {selected_count}")
            print(f"  Visual embeddings available: {vit_embeds.shape[0]}")
            
            if selected_count != vit_embeds.shape[0]:
                print(f"  ⚠️  MISMATCH: Expected {selected_count} visual embeddings, got {vit_embeds.shape[0]}")
                print(f"  This is the root cause of the tensor shape mismatch!")
                
                # Let's see what the actual tokens are
                print(f"\n🔍 Token analysis:")
                tokens = input_ids[0].tolist()  # First sample
                for i, token_id in enumerate(tokens):
                    token = tokenizer.decode([token_id])
                    if token_id == img_context_id:
                        print(f"    Position {i}: {token_id} -> '{token}' (IMG_CONTEXT)")
                    elif i < 10 or token_id == img_context_id:  # Show first 10 and all IMG_CONTEXT
                        print(f"    Position {i}: {token_id} -> '{token}'")
            
            # Try the forward pass anyway
            outputs = model(**forward_batch)
            print(f"✅ Forward pass successful despite warnings!")
            print(f"  Loss: {outputs.loss}")
            print(f"  Logits shape: {outputs.logits.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the debugging"""
    print("🔍 DEBUGGING VISUAL EMBEDDING INTEGRATION")
    print("=" * 60)
    
    success = debug_visual_embedding_mismatch()
    
    if success:
        print("\n✅ Analysis completed successfully!")
        print("The visual embedding mismatch has been identified.")
    else:
        print("\n❌ Analysis failed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())