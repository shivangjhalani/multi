#!/usr/bin/env python3
"""
Configuration Examples for Multimodal CoCoNuT

This script demonstrates various ways to create and customize configurations
for different training scenarios and use cases.

Usage:
    python examples/config_examples.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from multimodal_coconut import (
    Config,
    load_config,
    create_config_from_template,
    validate_config,
    print_config_summary,
    get_config_diff
)


def example_1_basic_config_creation():
    """Example 1: Basic configuration creation from templates"""
    print("="*60)
    print("EXAMPLE 1: Basic Configuration Creation")
    print("="*60)
    
    # Create different types of configurations
    configs = {}
    
    # Default configuration
    configs['default'] = create_config_from_template('default')
    print("✓ Created default configuration")
    
    # CoT pre-training configuration
    configs['cot'] = create_config_from_template('cot')
    print("✓ Created CoT pre-training configuration")
    
    # CoCoNuT training configuration
    configs['coconut'] = create_config_from_template('coconut')
    print("✓ Created CoCoNuT training configuration")
    
    # Evaluation configuration
    configs['eval'] = create_config_from_template('eval')
    print("✓ Created evaluation configuration")
    
    # Debug configuration
    configs['debug'] = create_config_from_template('debug')
    print("✓ Created debug configuration")
    
    # Print summaries
    for name, config in configs.items():
        print(f"\n--- {name.upper()} CONFIG ---")
        print(f"Name: {config.get('name', 'N/A')}")
        print(f"Mode: {'CoT' if config.get('cot', False) else 'CoCoNuT'}")
        print(f"Epochs: {config.get('num_epochs', 'N/A')}")
        print(f"Batch size: {config.get('batch_size_training', 'N/A')}")
        if config.get('coconut', False):
            print(f"C-thought: {config.get('c_thought', 'N/A')}")
            print(f"Max latent stage: {config.get('max_latent_stage', 'N/A')}")
    
    return configs


def example_2_custom_config_creation():
    """Example 2: Creating custom configurations with overrides"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration Creation")
    print("="*60)
    
    # Create custom configuration with specific parameters
    custom_config = create_config_from_template(
        'coconut',
        name="my-custom-experiment",
        c_thought=3,
        max_latent_stage=6,
        epochs_per_stage=3,
        batch_size_training=4,
        learning_rate=5e-6,
        image_size=224,
        max_num_patches=8,
        num_epochs=30
    )
    
    print("✓ Created custom configuration with overrides")
    print_config_summary(custom_config)
    
    # Save custom configuration
    custom_config.save('examples/custom_config.yaml')
    print("✓ Saved custom configuration to examples/custom_config.yaml")
    
    return custom_config


def example_3_config_validation():
    """Example 3: Configuration validation and error handling"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Configuration Validation")
    print("="*60)
    
    # Valid configuration
    valid_config = create_config_from_template('default')
    try:
        validate_config(valid_config)
        print("✓ Valid configuration passed validation")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid configuration examples
    invalid_configs = [
        # Missing required field
        Config({"name": "test"}),
        
        # Invalid c_thought
        Config({"model_id": "test", "c_thought": 0}),
        
        # Invalid batch size
        Config({"model_id": "test", "c_thought": 2, "batch_size_training": -1}),
        
        # Invalid learning rate
        Config({"model_id": "test", "c_thought": 2, "learning_rate": -0.1}),
    ]
    
    for i, config in enumerate(invalid_configs, 1):
        try:
            validate_config(config)
            print(f"✗ Invalid config {i} unexpectedly passed validation")
        except Exception as e:
            print(f"✓ Invalid config {i} correctly failed validation: {str(e)[:50]}...")


def example_4_config_manipulation():
    """Example 4: Configuration manipulation and merging"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Configuration Manipulation")
    print("="*60)
    
    # Create base configuration
    base_config = create_config_from_template('default')
    print("✓ Created base configuration")
    
    # Update configuration
    base_config.update(
        name="updated-experiment",
        learning_rate=2e-5,
        batch_size_training=16
    )
    print("✓ Updated configuration with new values")
    
    # Create another configuration
    override_config = Config({
        "c_thought": 4,
        "max_latent_stage": 8,
        "temperature": 0.8
    })
    
    # Merge configurations
    merged_config = base_config.merge(override_config)
    print("✓ Merged configurations")
    
    # Show differences
    print("\nConfiguration differences:")
    diff = get_config_diff(base_config, merged_config)
    for key, (old_val, new_val) in diff.items():
        print(f"  {key}: {old_val} → {new_val}")
    
    return merged_config


def example_5_environment_variables():
    """Example 5: Using environment variables in configurations"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Environment Variables")
    print("="*60)
    
    # Create configuration with environment variables
    config_with_env = {
        "name": "env-experiment",
        "model_id": "OpenGVLab/InternVL3-1B-Pretrained",
        "c_thought": 2,
        "train_data_path": "${DATA_ROOT:data/aokvqa}/train.json",
        "val_data_path": "${DATA_ROOT:data/aokvqa}/val.json",
        "image_root": "${DATA_ROOT:data/aokvqa}/images",
        "save_path": "${CHECKPOINT_DIR:checkpoints}",
        "batch_size_training": "${BATCH_SIZE:8}",
        "learning_rate": "${LEARNING_RATE:1e-5}"
    }
    
    # Save configuration with environment variables
    import yaml
    with open('examples/env_config.yaml', 'w') as f:
        yaml.dump(config_with_env, f, default_flow_style=False, indent=2)
    
    print("✓ Created configuration with environment variables")
    print("✓ Saved to examples/env_config.yaml")
    
    # Show how to set environment variables
    print("\nTo use this configuration, set environment variables:")
    print("export DATA_ROOT=/path/to/your/data")
    print("export CHECKPOINT_DIR=/path/to/checkpoints")
    print("export BATCH_SIZE=16")
    print("export LEARNING_RATE=5e-6")
    
    # Try loading (will use defaults if env vars not set)
    try:
        loaded_config = load_config('examples/env_config.yaml')
        print("✓ Successfully loaded configuration with environment variable substitution")
        print(f"  Data path: {loaded_config.train_data_path}")
        print(f"  Save path: {loaded_config.save_path}")
        print(f"  Batch size: {loaded_config.batch_size_training}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")


def example_6_stage_specific_configs():
    """Example 6: Stage-specific configuration updates"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Stage-Specific Configurations")
    print("="*60)
    
    from multimodal_coconut.config import update_config_for_stage
    
    # Base configuration
    base_config = create_config_from_template('coconut')
    print("✓ Created base CoCoNuT configuration")
    
    # Update for different stages
    stages = [0, 1, 2, 3, 4]
    
    for stage in stages:
        stage_config = update_config_for_stage(base_config, stage)
        mode = "CoT" if stage_config.get('cot', False) else "CoCoNuT"
        print(f"  Stage {stage}: {mode} mode")
    
    print("✓ Demonstrated stage-specific configuration updates")


def example_7_config_templates_comparison():
    """Example 7: Comparing different configuration templates"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Configuration Templates Comparison")
    print("="*60)
    
    templates = ['default', 'cot', 'coconut', 'eval', 'debug']
    configs = {}
    
    # Create all template configurations
    for template in templates:
        configs[template] = create_config_from_template(template)
    
    # Compare key parameters
    print("Template Comparison:")
    print(f"{'Parameter':<20} {'Default':<10} {'CoT':<10} {'CoCoNuT':<10} {'Eval':<10} {'Debug':<10}")
    print("-" * 80)
    
    key_params = [
        'num_epochs', 'batch_size_training', 'learning_rate', 
        'c_thought', 'max_latent_stage', 'only_eval'
    ]
    
    for param in key_params:
        row = f"{param:<20}"
        for template in templates:
            value = configs[template].get(param, 'N/A')
            row += f"{str(value):<10}"
        print(row)
    
    print("✓ Compared all configuration templates")


def main():
    """Run all configuration examples"""
    print("MULTIMODAL COCONUT CONFIGURATION EXAMPLES")
    print("This script demonstrates various configuration patterns and use cases.\n")
    
    try:
        # Run all examples
        example_1_basic_config_creation()
        example_2_custom_config_creation()
        example_3_config_validation()
        example_4_config_manipulation()
        example_5_environment_variables()
        example_6_stage_specific_configs()
        example_7_config_templates_comparison()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("- examples/custom_config.yaml")
        print("- examples/env_config.yaml")
        print("\nThese examples show how to:")
        print("1. Create configurations from templates")
        print("2. Customize configurations with overrides")
        print("3. Validate configuration parameters")
        print("4. Manipulate and merge configurations")
        print("5. Use environment variables")
        print("6. Handle stage-specific updates")
        print("7. Compare different templates")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()