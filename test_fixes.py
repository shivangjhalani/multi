#!/usr/bin/env python3
"""
Test the critical fixes for multimodal CoCoNuT
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

# Import all multimodal CoCoNuT components
from multimodal_coconut.config import Config
from multimodal_coconut.data.dataset import (
    MultimodalDataset, 
    MultimodalCollator,
    get_multimodal_dataset
)
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut


def setup_real_model_and_tokenizer():
    """Setup real InternVL model and tokenizer for testing"""
    try:
        model_id = "OpenGVLab/InternVL3-1B-Pretrained"
        
        print(f"Loading real InternVL model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Add CoCoNuT special tokens
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        # Get special token IDs
        start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        
        # Add missing attributes for compatibility
        tokenizer.latent_token_id = latent_id
        tokenizer.start_latent_id = start_latent_id
        tokenizer.end_latent_id = end_latent_id
        
        # Load base model (CPU only for testing)
        base_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        # Resize token embeddings
        old_vocab_size = base_model.language_model.config.vocab_size
        new_vocab_size = len(tokenizer)
        
        if new_vocab_size > old_vocab_size:
            base_model.language_model.resize_token_embeddings(new_vocab_size)
        
        # CRITICAL FIX: Set img_context_token_id properly
        try:
            img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            base_model.img_context_token_id = img_context_token_id
            print(f"✓ IMG_CONTEXT token ID set: {img_context_token_id}")
        except Exception as e:
            print(f"⚠️  Could not set IMG_CONTEXT token ID: {e}")
            base_model.img_context_token_id = None
        
        print(f"✓ Real model loaded successfully")
        print(f"  - Vocab size: {old_vocab_size} -> {new_vocab_size}")
        print(f"  - Special tokens: start={start_latent_id}, latent={latent_id}, end={end_latent_id}")
        
        return base_model, tokenizer
        
    except Exception as e:
        print(f"⚠️  Failed to load real model: {e}")
        return None, None


def create_test_data(temp_dir):
    """Create test dataset"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create test images
    image_paths = []
    for i in range(2):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"test_image_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    
    # Create training data
    train_data = []
    for i in range(2):
        sample = {
            "image_path": image_paths[i % len(image_paths)],
            "question": f"What is shown in this image number {i}?",
            "steps": [
                f"Step 1: I can see various elements in the image {i}",
                f"Step 2: The image shows specific visual details {i}",
            ],
            "answer": f"The answer is a detailed description of image {i}"
        }
        train_data.append(sample)
    
    # Save datasets
    train_path = Path(temp_dir) / "train.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    return str(train_path), str(images_dir)


def test_img_context_token_fix():
    """Test the IMG_CONTEXT token fix"""
    print("\n🔧 TESTING IMG_CONTEXT TOKEN FIX")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Setup model and tokenizer
        base_model, tokenizer = setup_real_model_and_tokenizer()
        
        if base_model is None or tokenizer is None:
            print("❌ Cannot test without real model")
            return False
        
        # Test tokenizer has IMG_CONTEXT token
        try:
            img_context_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            print(f"✓ IMG_CONTEXT token ID: {img_context_id}")
        except Exception as e:
            print(f"❌ IMG_CONTEXT token not found: {e}")
            return False
        
        # Create test data
        train_path, images_dir = create_test_data(temp_dir)
        
        # Create dataset with proper IMG_CONTEXT tokens
        dataset = get_multimodal_dataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_num_patches=4,
            max_size=2
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Check if IMG_CONTEXT tokens are in the tokenized data
        sample = dataset[0]
        input_ids = (
            sample['question_tokenized'] +
            list(sample['steps_tokenized'][0]) +
            sample['answer_tokenized']
        )
        
        has_img_context = img_context_id in input_ids
        print(f"✓ IMG_CONTEXT tokens in data: {has_img_context}")
        
        if not has_img_context:
            print("❌ IMG_CONTEXT tokens not found in tokenized data")
            return False
        
        # Create model
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print("✓ Model created successfully")
        
        # Create collator
        collator = MultimodalCollator(tokenizer=tokenizer, latent_id=tokenizer.latent_token_id)
        
        # Create batch
        features = [dataset[i] for i in range(len(dataset))]
        batch = collator(features)
        
        print(f"✓ Batch created successfully")
        print(f"  Batch keys: {list(batch.keys())}")
        
        # Test forward pass
        forward_batch = {k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']}
        
        print(f"✓ Forward batch prepared")
        print(f"  Forward batch keys: {list(forward_batch.keys())}")
        
        # Check if base model has img_context_token_id set
        print(f"✓ Base model img_context_token_id: {getattr(base_model, 'img_context_token_id', 'NOT SET')}")
        
        try:
            outputs = model(**forward_batch)
            print(f"✅ Forward pass successful!")
            print(f"  Loss: {outputs.loss}")
            print(f"  Logits shape: {outputs.logits.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Additional debugging
            print(f"\n🔍 Additional debugging:")
            print(f"  input_ids shape: {forward_batch['input_ids'].shape}")
            print(f"  pixel_values shape: {forward_batch['pixel_values'].shape}")
            print(f"  image_flags shape: {forward_batch['image_flags'].shape}")
            
            # Check for IMG_CONTEXT tokens in input_ids
            input_ids_flat = forward_batch['input_ids'].flatten()
            img_context_count = (input_ids_flat == img_context_id).sum().item()
            print(f"  IMG_CONTEXT tokens in batch: {img_context_count}")
            
            return False
        
    except Exception as e:
        print(f"💥 Critical test failure: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the fix test"""
    print("🧪 TESTING CRITICAL FIXES FOR MULTIMODAL COCONUT")
    print("=" * 60)
    
    success = test_img_context_token_fix()
    
    if success:
        print("\n🎉 CRITICAL FIXES WORKING!")
        print("The IMG_CONTEXT token issue has been resolved.")
    else:
        print("\n⚠️  Fixes still need work.")
        print("The IMG_CONTEXT token issue persists.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())