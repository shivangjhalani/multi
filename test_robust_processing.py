#!/usr/bin/env python3
"""
Test script to demonstrate robust dataset processing that prevents hanging
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path

# Add the multimodal_coconut directory to the path
sys.path.insert(0, 'multimodal_coconut')

import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

# Import our modules
from multimodal_coconut.data.dataset import MultimodalDataset, get_multimodal_dataset
from multimodal_coconut.data.image_processor import create_image_processor
from multimodal_coconut.data.dataset_utils import (
    RobustDatasetProcessor, 
    validate_dataset_sample, 
    estimate_memory_usage,
    ProcessingMonitor
)


def create_challenging_test_data():
    """Create test data that would typically cause dataset.map to hang"""
    temp_dir = tempfile.mkdtemp()
    
    test_data = []
    test_images = []
    
    print("Creating challenging test dataset...")
    
    # Create various types of problematic samples
    for i in range(20):
        if i % 5 == 0:
            # Large images that could cause memory issues
            img_array = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        elif i % 5 == 1:
            # Very small images
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        elif i % 5 == 2:
            # Extreme aspect ratios
            img_array = np.random.randint(0, 255, (100, 1000, 3), dtype=np.uint8)
        elif i % 5 == 3:
            # Normal images
            img_array = np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
        else:
            # Another normal size
            img_array = np.random.randint(0, 255, (224, 336, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        img_path = os.path.join(temp_dir, f'test_image_{i}.jpg')
        img.save(img_path, quality=95)
        test_images.append(img_path)
        
        # Create sample with varying complexity
        if i % 7 == 0:
            # Very long reasoning chains
            steps = [f"Step {j}: This is a detailed reasoning step with lots of text to make tokenization slower." for j in range(20)]
        elif i % 7 == 1:
            # Short reasoning
            steps = ["Quick analysis"]
        else:
            # Normal reasoning
            steps = [f"Step {j}: Analyzing the image content" for j in range(3)]
        
        sample = {
            "image_path": img_path,
            "question": f"Question {i}: What do you see in this image with index {i}?",
            "steps": steps,
            "answer": f"Answer {i}: This is test image number {i} with specific characteristics."
        }
        test_data.append(sample)
    
    # Add some samples with missing images (to test error handling)
    for i in range(3):
        sample = {
            "image_path": os.path.join(temp_dir, f'missing_image_{i}.jpg'),  # This file doesn't exist
            "question": f"Question about missing image {i}",
            "steps": ["Trying to analyze missing image"],
            "answer": f"This should handle missing image gracefully"
        }
        test_data.append(sample)
    
    # Save test data
    data_path = os.path.join(temp_dir, 'challenging_test_data.json')
    with open(data_path, 'w') as f:
        json.dump(test_data, f)
    
    print(f"Created {len(test_data)} challenging samples")
    return temp_dir, data_path


def test_memory_estimation():
    """Test memory estimation utility"""
    print("Testing memory estimation...")
    
    # Estimate memory for different dataset sizes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        memory_est = estimate_memory_usage(
            dataset_size=size,
            image_size=448,
            max_patches=12,
            avg_text_length=512
        )
        
        print(f"  Dataset size {size}:")
        print(f"    Per sample: {memory_est['per_sample_mb']:.1f} MB")
        print(f"    Total dataset: {memory_est['total_dataset_gb']:.2f} GB")
        print(f"    Image memory: {memory_est['image_memory_gb']:.2f} GB")
        print(f"    Text memory: {memory_est['text_memory_gb']:.2f} GB")
    
    return True


def test_robust_processing():
    """Test robust dataset processing that prevents hanging"""
    print("Testing robust dataset processing...")
    
    # Create challenging test data
    temp_dir, data_path = create_challenging_test_data()
    
    # Load tokenizer
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
        print("  Creating multimodal dataset with robust processing...")
        start_time = time.time()
        
        # This should NOT hang, even with challenging data
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_size=448,
            max_num_patches=12,
            max_size=25  # Process all samples including problematic ones
        )
        
        processing_time = time.time() - start_time
        print(f"  ✓ Dataset created successfully in {processing_time:.2f}s")
        print(f"  ✓ Dataset length: {len(dataset)}")
        
        # Validate samples
        valid_samples = 0
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            if validate_dataset_sample(sample):
                valid_samples += 1
            
            print(f"  ✓ Sample {i}: {list(sample.keys())}")
            print(f"    - Pixel values shape: {sample['pixel_values'].shape}")
            print(f"    - Question tokens: {len(sample['question_tokenized'])}")
            print(f"    - Steps: {len(sample['steps_tokenized'])}")
        
        print(f"  ✓ Valid samples: {valid_samples}/{min(5, len(dataset))}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Robust processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processing_monitor():
    """Test the processing monitor"""
    print("Testing ProcessingMonitor...")
    
    monitor = ProcessingMonitor(timeout_seconds=10)
    
    # Simulate processing
    monitor.start_monitoring(100)
    
    for i in range(10):
        time.sleep(0.1)  # Simulate work
        monitor.update_progress(i + 1)
    
    monitor.stop_monitoring()
    
    print("  ✓ ProcessingMonitor completed successfully")
    return True


def test_error_recovery():
    """Test error recovery mechanisms"""
    print("Testing error recovery...")
    
    # Create dataset with intentionally problematic samples
    temp_dir = tempfile.mkdtemp()
    
    problematic_data = [
        {
            "image_path": "/nonexistent/path/image.jpg",  # Missing file
            "question": "What's in this missing image?",
            "steps": ["Analyzing missing content"],
            "answer": "Should handle gracefully"
        },
        {
            "image_path": temp_dir,  # Directory instead of file
            "question": "What's in this directory?",
            "steps": ["Trying to load directory as image"],
            "answer": "Should create dummy sample"
        }
    ]
    
    # Add one valid sample
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    valid_img_path = os.path.join(temp_dir, 'valid.jpg')
    img.save(valid_img_path)
    
    problematic_data.append({
        "image_path": valid_img_path,
        "question": "What's in this valid image?",
        "steps": ["Analyzing valid content"],
        "answer": "This should work fine"
    })
    
    data_path = os.path.join(temp_dir, 'problematic_data.json')
    with open(data_path, 'w') as f:
        json.dump(problematic_data, f)
    
    # Test processing with errors
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        special_tokens = ["<image>", "<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_size=224,
            max_num_patches=6,
            max_size=10
        )
        
        print(f"  ✓ Error recovery successful, dataset length: {len(dataset)}")
        
        # Check that we got some samples (including dummy ones)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  ✓ Sample structure preserved: {list(sample.keys())}")
            return True
        else:
            print("  ✗ No samples in dataset")
            return False
            
    except Exception as e:
        print(f"  ✗ Error recovery failed: {e}")
        return False


def main():
    """Run all robust processing tests"""
    print("Testing Robust Multimodal Data Processing")
    print("=" * 60)
    print("This test demonstrates how our implementation prevents dataset.map from hanging")
    print()
    
    tests = [
        ("Memory Estimation", test_memory_estimation),
        ("Processing Monitor", test_processing_monitor),
        ("Error Recovery", test_error_recovery),
        ("Robust Processing", test_robust_processing),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}\n")
        except Exception as e:
            print(f"✗ FAILED with exception: {e}\n")
            results.append(False)
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("-" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All robust processing tests passed!")
        print("\nKey improvements that prevent hanging:")
        print("  • Conservative multiprocessing (max 8 processes)")
        print("  • Timeout protection for individual samples")
        print("  • Graceful error handling with dummy samples")
        print("  • Memory usage estimation and monitoring")
        print("  • Sequential fallback when parallel processing fails")
        print("  • Progress monitoring with hang detection")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())