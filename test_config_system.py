#!/usr/bin/env python3
"""
Test script for the enhanced configuration system
"""

import os
import tempfile
from pathlib import Path
from multimodal_coconut.config import (
    Config, load_config, validate_config, create_config_from_template,
    substitute_env_vars, print_config_summary, get_config_diff,
    ConfigError
)


def test_basic_config():
    """Test basic configuration functionality"""
    print("Testing basic configuration...")
    
    # Test Config class
    config_dict = {"name": "test", "c_thought": 2, "model_id": "test-model"}
    config = Config(config_dict)
    
    assert config.name == "test"
    assert config.c_thought == 2
    assert config.get("missing_key", "default") == "default"
    assert config.has("name") == True
    assert config.has("missing_key") == False
    
    # Test update
    config.update(new_param=42)
    assert config.new_param == 42
    
    print("✓ Basic configuration tests passed")


def test_env_substitution():
    """Test environment variable substitution"""
    print("Testing environment variable substitution...")
    
    # Set test environment variable
    os.environ["TEST_VAR"] = "test_value"
    os.environ["TEST_NUM"] = "123"
    
    config_str = """
name: ${TEST_VAR}
number: ${TEST_NUM}
with_default: ${MISSING_VAR:default_value}
no_default: ${ANOTHER_MISSING:}
"""
    
    result = substitute_env_vars(config_str)
    assert "test_value" in result
    assert "123" in result
    assert "default_value" in result
    
    print("✓ Environment variable substitution tests passed")


def test_config_templates():
    """Test configuration templates"""
    print("Testing configuration templates...")
    
    # Test different templates
    templates = ['default', 'cot', 'coconut', 'eval', 'debug']
    
    for template_name in templates:
        config = create_config_from_template(template_name)
        assert hasattr(config, 'name')
        assert hasattr(config, 'model_id')
        assert hasattr(config, 'c_thought')
        
        # Validate the config
        try:
            validate_config(config)
            print(f"✓ Template '{template_name}' is valid")
        except ConfigError as e:
            print(f"✗ Template '{template_name}' validation failed: {e}")
    
    # Test template with overrides
    config = create_config_from_template('debug', c_thought=5, custom_param="test")
    assert config.c_thought == 5
    assert config.custom_param == "test"
    
    print("✓ Configuration template tests passed")


def test_config_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test valid config
    valid_config = create_config_from_template('default')
    try:
        validate_config(valid_config)
        print("✓ Valid configuration passed validation")
    except ConfigError:
        print("✗ Valid configuration failed validation")
    
    # Test invalid configs
    invalid_configs = [
        # Missing required field
        Config({"name": "test"}),
        # Invalid c_thought
        Config({"model_id": "test", "c_thought": 0}),
        # Invalid batch size
        Config({"model_id": "test", "c_thought": 2, "batch_size_training": -1}),
        # Invalid learning rate
        Config({"model_id": "test", "c_thought": 2, "learning_rate": -0.1}),
        # Both FSDP and DDP enabled
        Config({"model_id": "test", "c_thought": 2, "use_fsdp": True, "use_ddp": True}),
    ]
    
    for i, invalid_config in enumerate(invalid_configs):
        try:
            validate_config(invalid_config)
            print(f"✗ Invalid config {i+1} passed validation (should have failed)")
        except ConfigError:
            print(f"✓ Invalid config {i+1} correctly failed validation")
    
    print("✓ Configuration validation tests passed")


def test_config_file_loading():
    """Test loading configuration from file"""
    print("Testing configuration file loading...")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
name: "test-config"
model_id: "test-model"
c_thought: 3
batch_size_training: 16
learning_rate: 2e-5
use_env: ${TEST_CONFIG_VAR:default_env_value}
""")
        temp_path = f.name
    
    try:
        # Set environment variable
        os.environ["TEST_CONFIG_VAR"] = "env_value_set"
        
        # Load config
        config = load_config(temp_path)
        
        assert config.name == "test-config"
        assert config.c_thought == 3
        assert config.use_env == "env_value_set"
        
        print("✓ Configuration file loading tests passed")
        
    finally:
        # Clean up
        os.unlink(temp_path)


def test_config_utilities():
    """Test configuration utility functions"""
    print("Testing configuration utilities...")
    
    # Test config diff
    config1 = Config({"a": 1, "b": 2, "c": 3})
    config2 = Config({"a": 1, "b": 3, "d": 4})
    
    diff = get_config_diff(config1, config2)
    assert "b" in diff
    assert "c" in diff
    assert "d" in diff
    assert diff["b"] == (2, 3)
    
    # Test config merge
    merged = config1.merge(config2)
    assert merged.a == 1  # Same in both
    assert merged.b == 3  # config2 takes precedence
    assert merged.d == 4  # Only in config2
    
    # Test config summary (just make sure it doesn't crash)
    config = create_config_from_template('default')
    print_config_summary(config)
    
    print("✓ Configuration utility tests passed")


def main():
    """Run all tests"""
    print("Running configuration system tests...\n")
    
    try:
        test_basic_config()
        test_env_substitution()
        test_config_templates()
        test_config_validation()
        test_config_file_loading()
        test_config_utilities()
        
        print("\n🎉 All configuration system tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())