#!/usr/bin/env python3
"""
Test script for multimodal data pipeline components
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the multimodal_coconut directory to the path
sys.path.insert(0, 'multimodal_coconut')

import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

# Import our modules
from multimodal_coconut.data.dataset import MultimodalDataset, MultimodalCollator, get_multimodal_dataset
from multimodal_coconut.data.image_processor import ImageProcessor, create_image_processor
from multimodal_coconut.data.dataset_utils import RobustDatasetProcessor, validate_dataset_sample, estimate_memory_usage


def create_test_data():
    """Create test data for validation"""
    # Create temporary directory for test images
    temp_dir = tempfile.mkdtemp()
    
    # Create test images
    test_images = []
    for i in range(3):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(temp_dir, f'test_image_{i}.jpg')
        img.save(img_path)
        test_images.append(img_path)
    
    # Create test JSON data
    test_data = [
        {
            "image_path": test_images[0],
            "question": "What do you see in this image?",
            "steps": [
                "I need to analyze the visual content",
                "The image shows various patterns and colors"
            ],
            "answer": "This is a test image with random patterns."
        },
        {
            "image_path": test_images[1],
            "question": "Describe the main features.",
            "steps": [
                "Looking at the image structure",
                "Identifying key visual elements"
            ],
            "answer": "The image contains randomly generated pixel values."
        },
        {
            "image_path": test_images[2],
            "question": "What colors are present?",
            "steps": [
                "Examining the color distribution"
            ],
            "answer": "Multiple colors are present in random arrangement."
        }
    ]
    
    # Save test data
    data_path = os.path.join(temp_dir, 'test_data.json')
    with open(data_path, 'w') as f:
        json.dump(test_data, f)
    
    return temp_dir, data_path


def test_image_processor():
    """Test the image processor"""
    print("Testing ImageProcessor...")
    
    # Create test image
    temp_dir = tempfile.mkdtemp()
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = os.path.join(temp_dir, 'test.jpg')
    img.save(img_path)
    
    # Test processor
    processor = create_image_processor(image_size=224, max_num_patches=6)
    
    # Load and process image
    pixel_values = processor.load_image(img_path)
    
    print(f"  ✓ Image processed successfully")
    print(f"  ✓ Output shape: {pixel_values.shape}")
    print(f"  ✓ Number of patches: {pixel_values.shape[0]}")
    
    # Test error handling
    try:
        processor.load_image("nonexistent.jpg")
        print(f"  ✓ Error handling works (dummy tensor returned)")
    except:
        print(f"  ✗ Error handling failed")
    
    # Print stats
    stats = processor.get_stats()
    print(f"  ✓ Processing stats: {stats}")
    
    return True


def test_multimodal_dataset():
    """Test the multimodal dataset"""
    print("Testing MultimodalDataset...")
    
    # Create test data
    temp_dir, data_path = create_test_data()
    
    # Load tokenizer (using a simple tokenizer for testing)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        special_tokens = ["<image>", "<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
    except Exception as e:
        print(f"  ✗ Could not load tokenizer: {e}")
        return False
    
    try:
        # Create dataset
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_size=224,
            max_num_patches=6,
            max_size=10
        )
        
        print(f"  ✓ Dataset created successfully")
        print(f"  ✓ Dataset length: {len(dataset)}")
        
        # Test sample access
        sample = dataset[0]
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        print(f"  ✓ Pixel values shape: {sample['pixel_values'].shape}")
        print(f"  ✓ Question tokens length: {len(sample['question_tokenized'])}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_multimodal_collator():
    """Test the multimodal collator"""
    print("Testing MultimodalCollator...")
    
    try:
        # Create test data
        temp_dir, data_path = create_test_data()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens
        special_tokens = ["<image>", "<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        
        # Create dataset
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_size=224,
            max_num_patches=6,
            max_size=3
        )
        
        # Create collator
        collator = MultimodalCollator(tokenizer=tokenizer, latent_id=latent_id)
        
        # Test collation
        batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
        
        print(f"  ✓ Batch created successfully")
        print(f"  ✓ Batch keys: {list(batch.keys())}")
        print(f"  ✓ Input IDs shape: {batch['input_ids'].shape}")
        print(f"  ✓ Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  ✓ Number of patches per sample: {batch['num_patches_list']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Collator test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Running Multimodal Data Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_image_processor,
        test_multimodal_dataset,
        test_multimodal_collator
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    print("Test Summary")
    print("-" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())