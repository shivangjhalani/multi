#!/usr/bin/env python3
"""
Simple test to verify the multimodal data pipeline works without hanging
"""

import sys
import os
import tempfile
import json
from PIL import Image
import numpy as np

# Add path
sys.path.insert(0, 'multimodal_coconut')

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("Testing basic multimodal data pipeline...")
    
    # Test image processor
    try:
        from multimodal_coconut.data.image_processor import ImageProcessor
        
        # Create test image
        temp_dir = tempfile.mkdtemp()
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = os.path.join(temp_dir, 'test.jpg')
        img.save(img_path)
        
        # Test processor
        processor = ImageProcessor(image_size=224, max_num_patches=4)
        pixel_values = processor.load_image(img_path)
        
        print(f"✓ ImageProcessor works: {pixel_values.shape}")
        
        # Test error handling
        dummy_tensor = processor.load_image("nonexistent.jpg")
        print(f"✓ Error handling works: {dummy_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def test_dataset_utils():
    """Test dataset utilities"""
    print("Testing dataset utilities...")
    
    try:
        from multimodal_coconut.data.dataset_utils import estimate_memory_usage, ProcessingMonitor
        
        # Test memory estimation
        memory_est = estimate_memory_usage(100, 448, 12, 512)
        print(f"✓ Memory estimation: {memory_est['total_dataset_gb']:.2f} GB for 100 samples")
        
        # Test monitor
        monitor = ProcessingMonitor(timeout_seconds=5)
        monitor.start_monitoring(10)
        monitor.update_progress(5)
        monitor.stop_monitoring()
        print("✓ ProcessingMonitor works")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset utils test failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("Simple Multimodal Data Pipeline Test")
    print("=" * 40)
    
    tests = [test_basic_functionality, test_dataset_utils]
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test failed: {e}")
            results.append(False)
        print()
    
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())