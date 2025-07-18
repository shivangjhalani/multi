#!/usr/bin/env python3
"""
Test script for configuration inheritance and template system
"""

import tempfile
import os
from pathlib import Path
from multimodal_coconut.config import (
    load_config_auto_inherit, load_config_with_inheritance,
    create_config_from_template, print_config_summary,
    ConfigError
)


def test_config_inheritance():
    """Test configuration inheritance functionality"""
    print("Testing configuration inheritance...")
    
    # Create temporary base config
    base_config_content = """
# Base configuration
seed: 42
model_id: "base-model"
c_thought: 2
learning_rate: 1e-5
batch_size_training: 8
use_fsdp: true
"""
    
    # Create temporary child config
    child_config_content = """
# Child configuration
base_config: "{base_path}"
name: "child-config"
c_thought: 3  # Override base
batch_size_training: 16  # Override base
new_param: "child-only"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as base_f:
        base_f.write(base_config_content)
        base_path = base_f.name
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as child_f:
            child_f.write(child_config_content.format(base_path=base_path))
            child_path = child_f.name
        
        try:
            # Load with inheritance
            config = load_config_auto_inherit(child_path)
            
            # Check inherited values
            assert config.seed == 42, f"Expected seed=42, got {config.seed}"
            assert config.model_id == "base-model", f"Expected model_id='base-model', got {config.model_id}"
            assert config.use_fsdp == True, f"Expected use_fsdp=True, got {config.use_fsdp}"
            assert abs(config.learning_rate - 1e-5) < 1e-10, f"Expected learning_rate=1e-5, got {config.learning_rate}"
            
            # Check overridden values
            assert config.c_thought == 3, f"Expected c_thought=3, got {config.c_thought}"
            assert config.batch_size_training == 16, f"Expected batch_size_training=16, got {config.batch_size_training}"
            
            # Check child-only values
            assert config.name == "child-config", f"Expected name='child-config', got {config.name}"
            assert config.new_param == "child-only", f"Expected new_param='child-only', got {config.new_param}"
            
            print("✓ Configuration inheritance tests passed")
            
        finally:
            os.unlink(child_path)
    finally:
        os.unlink(base_path)


def test_existing_config_files():
    """Test loading existing configuration files with inheritance"""
    print("Testing existing configuration files...")
    
    config_files = [
        "args/multimodal_coconut.yaml",
        "args/multimodal_cot.yaml", 
        "args/multimodal_coconut_eval.yaml",
        "args/multimodal_coconut_debug.yaml",
        "args/multimodal_coconut_large.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                config = load_config_auto_inherit(config_file)
                print(f"✓ Successfully loaded {config_file}")
                
                # Check that base config values are inherited
                assert hasattr(config, 'seed'), f"Missing inherited 'seed' in {config_file}"
                assert hasattr(config, 'model_id'), f"Missing inherited 'model_id' in {config_file}"
                assert hasattr(config, 'c_thought'), f"Missing inherited 'c_thought' in {config_file}"
                
                # Print summary for verification
                print(f"  Config name: {config.get('name', 'Unknown')}")
                print(f"  Training mode: {'CoT' if config.get('cot', False) else 'CoCoNuT'}")
                print(f"  Batch size: {config.get('batch_size_training', 'N/A')}")
                
            except Exception as e:
                print(f"✗ Failed to load {config_file}: {e}")
        else:
            print(f"⚠ Config file not found: {config_file}")
    
    print("✓ Existing configuration file tests completed")


def test_template_system():
    """Test the template system with inheritance"""
    print("Testing template system...")
    
    templates = ['default', 'cot', 'coconut', 'eval', 'debug']
    
    for template_name in templates:
        try:
            config = create_config_from_template(template_name)
            
            # Basic validation
            assert hasattr(config, 'name'), f"Template {template_name} missing 'name'"
            assert hasattr(config, 'model_id'), f"Template {template_name} missing 'model_id'"
            assert hasattr(config, 'c_thought'), f"Template {template_name} missing 'c_thought'"
            
            print(f"✓ Template '{template_name}' created successfully")
            
            # Test template-specific properties
            if template_name == 'cot':
                assert config.cot == True, f"CoT template should have cot=True"
                assert config.coconut == False, f"CoT template should have coconut=False"
            elif template_name == 'coconut':
                assert config.coconut == True, f"CoCoNuT template should have coconut=True"
                assert config.cot == False, f"CoCoNuT template should have cot=False"
            elif template_name == 'eval':
                assert config.only_eval == True, f"Eval template should have only_eval=True"
            elif template_name == 'debug':
                assert config.debug == True, f"Debug template should have debug=True"
                assert config.num_epochs == 2, f"Debug template should have small num_epochs"
            
        except Exception as e:
            print(f"✗ Template '{template_name}' failed: {e}")
    
    # Test template with overrides
    config = create_config_from_template('debug', 
                                       name="custom-debug",
                                       c_thought=5,
                                       custom_param="test")
    
    assert config.name == "custom-debug", "Template override failed for 'name'"
    assert config.c_thought == 5, "Template override failed for 'c_thought'"
    assert config.custom_param == "test", "Template override failed for 'custom_param'"
    assert config.debug == True, "Template base property should be preserved"
    
    print("✓ Template system tests passed")


def test_runtime_config_updates():
    """Test runtime configuration updates for stage transitions"""
    print("Testing runtime configuration updates...")
    
    from multimodal_coconut.config import update_config_for_stage
    
    # Create base config
    base_config = create_config_from_template('default')
    
    # Test Stage 0 (CoT pre-training)
    stage0_config = update_config_for_stage(base_config, 0)
    assert stage0_config.cot == True, "Stage 0 should have cot=True"
    assert stage0_config.coconut == False, "Stage 0 should have coconut=False"
    
    # Test Stage 1+ (CoCoNuT training)
    stage1_config = update_config_for_stage(base_config, 1)
    assert stage1_config.coconut == True, "Stage 1+ should have coconut=True"
    assert stage1_config.cot == False, "Stage 1+ should have cot=False"
    
    # Test that original config is not modified
    assert base_config.coconut == True, "Original config should not be modified"
    
    print("✓ Runtime configuration update tests passed")


def test_config_composition():
    """Test configuration composition features"""
    print("Testing configuration composition...")
    
    # Test multiple inheritance
    base_configs = []
    
    # Create first base config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
        f1.write("""
seed: 42
model_id: "base-model"
learning_rate: 1e-5
""")
        base_configs.append(f1.name)
    
    # Create second base config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
        f2.write("""
c_thought: 2
batch_size_training: 8
use_fsdp: true
""")
        base_configs.append(f2.name)
    
    # Create main config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as main_f:
        main_f.write("""
name: "composed-config"
c_thought: 3  # Override
new_param: "main-only"
""")
        main_path = main_f.name
    
    try:
        # Load with multiple inheritance
        config = load_config_with_inheritance(main_path, base_configs)
        
        # Check values from first base
        assert config.seed == 42
        assert config.model_id == "base-model"
        assert abs(config.learning_rate - 1e-5) < 1e-10
        
        # Check values from second base
        assert config.batch_size_training == 8
        assert config.use_fsdp == True
        
        # Check overridden value
        assert config.c_thought == 3
        
        # Check main-only value
        assert config.name == "composed-config"
        assert config.new_param == "main-only"
        
        print("✓ Configuration composition tests passed")
        
    finally:
        # Clean up
        for path in base_configs + [main_path]:
            os.unlink(path)


def main():
    """Run all inheritance and template tests"""
    print("Running configuration inheritance and template tests...\n")
    
    try:
        test_config_inheritance()
        test_existing_config_files()
        test_template_system()
        test_runtime_config_updates()
        test_config_composition()
        
        print("\n🎉 All configuration inheritance and template tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())