#!/usr/bin/env python3
"""
Test script showing how to properly create Config objects for multimodal CoCoNuT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_coconut.config import Config, create_default_config

def test_config_creation():
    """Test different ways to create Config objects"""
    
    print("=== Testing Config Creation ===")
    
    # Method 1: Using create_default_config()
    print("\n1. Creating config from default dictionary:")
    default_dict = create_default_config()
    config = Config(default_dict)
    print(f"✓ Config created successfully")
    print(f"  Model ID: {config.model_id}")
    print(f"  C-thought: {config.c_thought}")
    print(f"  Batch size: {config.batch_size_training}")
    
    # Method 2: Creating minimal config for testing
    print("\n2. Creating minimal config for testing:")
    minimal_config_dict = {
        "model_id": "OpenGVLab/InternVL3-1B-Pretrained",
        "c_thought": 2,
        "max_latent_stage": 4,
        "batch_size_training": 8,
        "learning_rate": 1e-5,
        "coconut": True,
        "cot": False
    }
    minimal_config = Config(minimal_config_dict)
    print(f"✓ Minimal config created successfully")
    print(f"  Model ID: {minimal_config.model_id}")
    print(f"  CoCoNuT enabled: {minimal_config.coconut}")
    
    # Method 3: Creating config and updating it
    print("\n3. Creating config and updating attributes:")
    config.update(
        batch_size_training=16,
        learning_rate=2e-5,
        custom_param="test_value"
    )
    print(f"✓ Config updated successfully")
    print(f"  New batch size: {config.batch_size_training}")
    print(f"  New learning rate: {config.learning_rate}")
    print(f"  Custom param: {config.custom_param}")
    
    return config

def create_test_config():
    """Create a simple config for testing the model"""
    config_dict = {
        "model_id": "OpenGVLab/InternVL3-1B-Pretrained",
        "c_thought": 2,
        "max_latent_stage": 4,
        "batch_size_training": 8,
        "batch_size_eval": 16,
        "learning_rate": 1e-5,
        "coconut": True,
        "cot": False,
        "image_size": 448,
        "use_flash_attn": True,
        "load_model_path": "None"
    }
    return Config(config_dict)

if __name__ == "__main__":
    # Test config creation
    config = test_config_creation()
    
    # Create a test config for model creation
    print("\n=== Creating Test Config for Model ===")
    test_config = create_test_config()
    print(f"✓ Test config ready for model creation")
    print(f"  Model: {test_config.model_id}")
    print(f"  Image size: {test_config.image_size}")
    print(f"  Flash attention: {test_config.use_flash_attn}")
    
    # Show how to use it with the model creation function
    print("\n=== Usage Example ===")
    print("# To create a model, use:")
    print("from multimodal_coconut.model.multimodal_coconut import create_multimodal_coconut_model")
    print("from multimodal_coconut.config import Config")
    print("")
    print("config_dict = {")
    print("    'model_id': 'OpenGVLab/InternVL3-1B-Pretrained',")
    print("    'c_thought': 2,")
    print("    'use_flash_attn': True")
    print("}")
    print("config = Config(config_dict)")
    print("model, tokenizer = create_multimodal_coconut_model(config)")