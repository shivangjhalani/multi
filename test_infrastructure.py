#!/usr/bin/env python3
"""
Test script to verify the multimodal CoCoNuT infrastructure is working correctly.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from multimodal_coconut import (
            Config,
            load_config,
            validate_config,
            set_seed,
            setup_logging,
            get_logger
        )
        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False
    
    try:
        from multimodal_coconut.utils import (
            init_distributed_training,
            is_main_process,
            count_parameters,
            get_device
        )
        print("✓ Utility imports successful")
    except ImportError as e:
        print(f"✗ Utility import failed: {e}")
        return False
    
    try:
        from multimodal_coconut.config import create_default_config
        print("✓ Config imports successful")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading and validation."""
    print("\nTesting configuration...")
    
    try:
        from multimodal_coconut import load_config, validate_config
        
        # Test loading the default config
        config = load_config("args/multimodal_coconut.yaml")
        print("✓ Configuration loaded successfully")
        
        # Test validation
        validate_config(config)
        print("✓ Configuration validation successful")
        
        # Test config access
        print(f"  Model ID: {config.model_id}")
        print(f"  CoCoNuT enabled: {config.coconut}")
        print(f"  C-thought: {config.c_thought}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utilities...")
    
    try:
        from multimodal_coconut.utils import set_seed, get_device
        from multimodal_coconut.utils.misc import format_time
        
        # Test seed setting
        set_seed(42)
        print("✓ Seed setting successful")
        
        # Test device detection
        device = get_device()
        print(f"✓ Device detection successful: {device}")
        
        # Test time formatting
        time_str = format_time(3661)
        print(f"✓ Time formatting successful: {time_str}")
        
        return True
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def test_logging():
    """Test logging setup."""
    print("\nTesting logging...")
    
    try:
        from multimodal_coconut.utils import setup_logging, get_logger
        
        # Setup logging (without W&B for testing)
        logger = setup_logging(use_wandb=False)
        print("✓ Logging setup successful")
        
        # Test logging
        logger.info("Test log message")
        print("✓ Logging test successful")
        
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Multimodal CoCoNuT Infrastructure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_utilities,
        test_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Infrastructure is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())