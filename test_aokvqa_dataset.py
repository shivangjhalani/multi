#!/usr/bin/env python3
"""
Test script for A-OKVQA dataset integration with multimodal CoCoNuT

This script tests the multimodal dataset implementation with the prepared A-OKVQA data
to ensure everything works correctly before proceeding with training.
"""

import sys
import json
from pathlib import Path

# Add the multimodal_coconut package to path
sys.path.append('.')

try:
    from transformers import AutoTokenizer
    from multimodal_coconut.data.dataset import MultimodalDataset, get_multimodal_dataset
    from multimodal_coconut.data.image_processor import ImageProcessor
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


def test_data_loading():
    """Test basic data loading"""
    print("Testing data loading...")
    
    # Check if data files exist
    data_dir = Path("data/aokvqa")
    train_file = data_dir / "train.json"
    val_file = data_dir / "validation.json"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return False
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return False
    
    # Load and check data structure
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    print(f"✅ Loaded {len(train_data)} training samples")
    
    # Check sample structure
    sample = train_data[0]
    required_fields = ['image_path', 'question', 'steps', 'answer']
    
    for field in required_fields:
        if field not in sample:
            print(f"❌ Missing field: {field}")
            return False
    
    print("✅ Data structure is correct")
    print(f"Sample question: {sample['question'][:100]}...")
    print(f"Sample steps: {len(sample['steps'])} steps")
    print(f"Sample answer: {sample['answer']}")
    
    return True


def test_image_processor():
    """Test image processor"""
    print("\nTesting image processor...")
    
    try:
        processor = ImageProcessor(
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True
        )
        
        # Test with first image
        data_dir = Path("data/aokvqa")
        with open(data_dir / "validation.json", 'r') as f:
            val_data = json.load(f)
        
        sample = val_data[0]
        image_path = data_dir / sample['image_path']
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return False
        
        # Process image
        pixel_values = processor.load_image(str(image_path))
        
        print(f"✅ Image processed successfully")
        print(f"Image tensor shape: {pixel_values.shape}")
        print(f"Number of patches: {pixel_values.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False


def test_tokenizer():
    """Test tokenizer loading"""
    print("\nTesting tokenizer...")
    
    try:
        # Load InternVL3 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        
        # Add special tokens for CoCoNuT
        special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"Vocabulary size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "What is in the image? The choices are 0: cat, 1: dog."
        tokens = tokenizer.encode(test_text)
        print(f"Test tokenization: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False


def test_multimodal_dataset():
    """Test multimodal dataset creation"""
    print("\nTesting multimodal dataset...")
    
    if not IMPORTS_OK:
        print("❌ Cannot test dataset - imports failed")
        return False
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        # Create dataset with small subset for testing
        data_path = "data/aokvqa/validation.json"
        image_root = "data/aokvqa"
        
        print("Creating multimodal dataset...")
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=image_root,
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=5  # Only process 5 samples for testing
        )
        
        print(f"✅ Dataset created successfully")
        print(f"Dataset size: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        
        # Check sample structure
        required_fields = ['pixel_values', 'question_tokenized', 'steps_tokenized', 'answer_tokenized', 'num_patches']
        for field in required_fields:
            if field not in sample:
                print(f"❌ Missing processed field: {field}")
                return False
        
        print(f"✅ Sample structure correct")
        print(f"Question tokens: {len(sample['question_tokenized'])}")
        print(f"Steps: {len(sample['steps_tokenized'])} reasoning steps")
        print(f"Answer tokens: {len(sample['answer_tokenized'])}")
        print(f"Image patches: {sample['num_patches']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multimodal dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_verification():
    """Test dataset verification function"""
    print("\nTesting dataset verification...")
    
    if not IMPORTS_OK:
        print("❌ Cannot test verification - imports failed")
        return False
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-Pretrained", trust_remote_code=True)
        special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        # Create dataset
        multimodal_dataset = MultimodalDataset(
            data_path="data/aokvqa/validation.json",
            tokenizer=tokenizer,
            image_root="data/aokvqa",
            image_size=448,
            max_num_patches=12,
            use_thumbnail=True,
            max_size=3  # Small subset
        )
        
        # Test verification
        result = multimodal_dataset.verify_sample(0)
        
        if result:
            print("✅ Dataset verification passed")
        else:
            print("❌ Dataset verification failed")
        
        return result
        
    except Exception as e:
        print(f"❌ Verification test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing A-OKVQA Dataset Integration")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Image Processor", test_image_processor),
        ("Tokenizer", test_tokenizer),
        ("Multimodal Dataset", test_multimodal_dataset),
        ("Dataset Verification", test_dataset_verification),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! A-OKVQA dataset integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)