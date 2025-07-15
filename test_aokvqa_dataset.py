#!/usr/bin/env python3
"""
Test script for A-OKVQA dataset integration with Multimodal CoCoNuT

This script tests the complete pipeline:
1. A-OKVQA dataset preparation
2. Multimodal dataset loading
3. Image processing
4. Tokenization
5. Data collation
6. Integration with CoCoNuT training format

Usage:
    python test_aokvqa_dataset.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from multimodal_coconut.data.dataset import MultimodalDataset, MultimodalCollator
from multimodal_coconut.data.image_processor import ImageProcessor
from multimodal_coconut.data.prepare_aokvqa import main as prepare_aokvqa_main


def test_aokvqa_preparation():
    """Test A-OKVQA dataset preparation"""
    print("🧪 Testing A-OKVQA Dataset Preparation")
    print("=" * 50)
    
    # Prepare small sample of validation data
    output_dir = Path("data/aokvqa")
    
    if not (output_dir / "validation.json").exists():
        print("📥 Preparing A-OKVQA validation data...")
        
        # Mock command line arguments for preparation script
        import sys
        original_argv = sys.argv
        sys.argv = [
            "prepare_aokvqa.py",
            "--output_dir", str(output_dir),
            "--splits", "validation",
            "--max_samples", "10"
        ]
        
        try:
            prepare_aokvqa_main()
        except SystemExit:
            pass  # Script calls sys.exit, which is normal
        finally:
            sys.argv = original_argv
    
    # Verify data exists
    data_file = output_dir / "validation.json"
    if not data_file.exists():
        print("❌ Failed to prepare A-OKVQA data")
        return False
    
    # Load and verify data structure
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("❌ No data found in prepared file")
        return False
    
    print(f"✅ Successfully prepared {len(data)} samples")
    
    # Check sample structure
    sample = data[0]
    required_fields = ['image_path', 'question', 'steps', 'answer']
    
    for field in required_fields:
        if field not in sample:
            print(f"❌ Missing required field: {field}")
            return False
    
    print("✅ Data structure is correct")
    print(f"Sample question: {sample['question'][:80]}...")
    print(f"Sample steps: {len(sample['steps'])} steps")
    print(f"Sample answer: {sample['answer']}")
    
    return True


def test_image_processor():
    """Test image processing functionality"""
    print("\n🔍 Image Processor")
    print("-" * 30)
    
    # Test with a sample image from the dataset
    output_dir = Path("data/aokvqa")
    data_file = output_dir / "validation.json"
    
    if not data_file.exists():
        print("❌ No validation data found")
        return False
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("❌ No samples in validation data")
        return False
    
    # Get first sample
    sample = data[0]
    image_path = output_dir / sample['image_path']
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Test image processor
    processor = ImageProcessor(
        image_size=448,
        max_num_patches=12,
        use_thumbnail=True
    )
    
    try:
        pixel_values = processor.load_image(str(image_path))
        print("✅ Image processed successfully")
        print(f"Image tensor shape: {pixel_values.shape}")
        print(f"Number of patches: {pixel_values.shape[0]}")
        
        # Verify tensor properties
        assert pixel_values.dim() == 4, f"Expected 4D tensor, got {pixel_values.dim()}D"
        assert pixel_values.shape[1:] == (3, 448, 448), f"Expected (3, 448, 448), got {pixel_values.shape[1:]}"
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer functionality"""
    print("\n🔍 Tokenizer")
    print("-" * 30)
    
    try:
        print("Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        
        # Add special tokens for CoCoNuT
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<image>"]
        tokenizer.add_tokens(special_tokens)
        
        print("✅ Tokenizer loaded successfully")
        print(f"Vocabulary size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "<image>\nWhat is in this image?\n"
        tokens = tokenizer.encode(test_text, add_special_tokens=True)
        print(f"Test tokenization: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False


def test_multimodal_dataset():
    """Test multimodal dataset functionality"""
    print("\n🔍 Multimodal Dataset")
    print("-" * 30)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<image>"]
        tokenizer.add_tokens(special_tokens)
        
        # Create dataset
        print("Creating multimodal dataset...")
        dataset = MultimodalDataset(
            data_path="data/aokvqa/validation.json",
            tokenizer=tokenizer,
            image_root="data/aokvqa",
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=5  # Small sample for testing
        )
        
        print("✅ Dataset created successfully")
        print(f"Dataset size: {len(dataset)}")
        
        # Test sample access
        sample = dataset[0]
        expected_keys = ['idx', 'pixel_values', 'question_tokenized', 'steps_tokenized', 'answer_tokenized', 'num_patches']
        
        print(f"Sample keys: {list(sample.keys())}")
        
        for key in expected_keys:
            if key not in sample:
                print(f"❌ Missing key: {key}")
                return False
        
        print("✅ Sample structure correct")
        print(f"Question tokens: {len(sample['question_tokenized'])}")
        print(f"Steps: {len(sample['steps_tokenized'])} reasoning steps")
        print(f"Answer tokens: {len(sample['answer_tokenized'])}")
        print(f"Image patches: {sample['num_patches']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal dataset test failed: {e}")
        return False


def test_dataset_verification():
    """Test dataset verification functionality"""
    print("\n🔍 Dataset Verification")
    print("-" * 30)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<image>"]
        tokenizer.add_tokens(special_tokens)
        
        # Create small dataset for verification
        dataset = MultimodalDataset(
            data_path="data/aokvqa/validation.json",
            tokenizer=tokenizer,
            image_root="data/aokvqa",
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=3
        )
        
        # Test verification
        success = dataset.verify_sample(0)
        
        if success:
            print("✅ Dataset verification passed")
            return True
        else:
            print("❌ Dataset verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Dataset verification test failed: {e}")
        return False


def test_data_collator():
    """Test multimodal data collator"""
    print("\n🔍 Data Collator")
    print("-" * 30)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<image>"]
        tokenizer.add_tokens(special_tokens)
        
        # Get latent token ID
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        
        # Create dataset
        dataset = MultimodalDataset(
            data_path="data/aokvqa/validation.json",
            tokenizer=tokenizer,
            image_root="data/aokvqa",
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=3
        )
        
        # Create collator
        collator = MultimodalCollator(
            tokenizer=tokenizer,
            latent_id=latent_id
        )
        
        # Test collation
        samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(samples)
        
        print("✅ Data collation successful")
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Number of patches: {batch['num_patches_list']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collator test failed: {e}")
        return False


def test_coconut_integration():
    """Test integration with CoCoNuT training format"""
    print("\n🔍 CoCoNuT Integration")
    print("-" * 30)
    
    try:
        from multimodal_coconut.data.dataset import get_multimodal_cot_latent_dataset
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<image>"]
        tokenizer.add_tokens(special_tokens)
        
        # Get special token IDs
        start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        
        # Create base dataset
        base_dataset = MultimodalDataset(
            data_path="data/aokvqa/validation.json",
            tokenizer=tokenizer,
            image_root="data/aokvqa",
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=3
        ).dataset
        
        # Mock config object
        class MockConfig:
            def __init__(self):
                self.uniform_prob = 0.1
                self.max_latent_stage = 2
                self.pad_latent_to_max = False
                self.no_cot = False
                self.c_thought = 2
        
        config = MockConfig()
        
        # Test CoT dataset preparation (Stage 0)
        cot_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=0,
            base_dataset=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id
        )
        
        print("✅ CoT dataset preparation successful")
        print(f"CoT dataset size: {len(cot_dataset)}")
        
        # Test CoCoNuT dataset preparation (Stage 1)
        coconut_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=1,
            base_dataset=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id
        )
        
        print("✅ CoCoNuT dataset preparation successful")
        print(f"CoCoNuT dataset size: {len(coconut_dataset)}")
        
        # Check sample structure
        sample = coconut_dataset[0]
        expected_keys = ['pixel_values', 'num_patches', 'input_ids', 'labels', 'attention_mask', 'idx', 'position_ids']
        
        for key in expected_keys:
            if key not in sample:
                print(f"❌ Missing key in CoCoNuT sample: {key}")
                return False
        
        print("✅ CoCoNuT sample structure correct")
        
        # Check for latent tokens in input
        input_ids = sample['input_ids']
        has_latent = latent_id in input_ids
        print(f"Contains latent tokens: {has_latent}")
        
        return True
        
    except Exception as e:
        print(f"❌ CoCoNuT integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing A-OKVQA Dataset Integration")
    print("=" * 50)
    
    tests = [
        ("Data Preparation", test_aokvqa_preparation),
        ("Image Processor", test_image_processor),
        ("Tokenizer", test_tokenizer),
        ("Multimodal Dataset", test_multimodal_dataset),
        ("Dataset Verification", test_dataset_verification),
        ("Data Collator", test_data_collator),
        ("CoCoNuT Integration", test_coconut_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed", end="")
    
    if passed == total:
        print(" 🎉")
        print("\nAll tests passed! A-OKVQA dataset integration is working correctly.")
    else:
        print(" ⚠️")
        print("\nSome tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)