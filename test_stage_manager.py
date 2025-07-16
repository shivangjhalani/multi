#!/usr/bin/env python3
"""
Test script for the multimodal CoCoNuT stage management system.

This script validates that the stage manager correctly implements the original
CoCoNuT curriculum progression and handles all edge cases properly.
"""

import sys
import os
sys.path.append('.')

from multimodal_coconut.config import Config
from multimodal_coconut.training.stage_manager import StageManager, create_stage_manager


def test_basic_stage_progression():
    """Test basic stage progression following original CoCoNuT pattern"""
    print("Testing basic stage progression...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 4,
        'c_thought': 2,
        'uniform_prob': 0.0,  # Disable for predictable testing
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Test stage calculation
    test_cases = [
        (0, 0),   # Epoch 0 -> Stage 0
        (4, 0),   # Epoch 4 -> Stage 0
        (5, 1),   # Epoch 5 -> Stage 1
        (9, 1),   # Epoch 9 -> Stage 1
        (10, 2),  # Epoch 10 -> Stage 2
        (15, 3),  # Epoch 15 -> Stage 3
        (20, 4),  # Epoch 20 -> Stage 4
        (25, 5),  # Epoch 25 -> Stage 5
    ]
    
    for epoch, expected_stage in test_cases:
        actual_stage = stage_manager.get_current_stage(epoch)
        assert actual_stage == expected_stage, f"Epoch {epoch}: expected stage {expected_stage}, got {actual_stage}"
        print(f"  ✓ Epoch {epoch} -> Stage {actual_stage}")
    
    print("  ✓ Basic stage progression test passed!")


def test_cot_mode():
    """Test CoT mode (should always return stage 0)"""
    print("Testing CoT mode...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 4,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': True,  # Force CoT mode
        'coconut': False,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # All epochs should return stage 0 in CoT mode
    for epoch in [0, 5, 10, 15, 20, 100]:
        stage = stage_manager.get_current_stage(epoch)
        assert stage == 0, f"CoT mode should always return stage 0, got {stage} for epoch {epoch}"
        print(f"  ✓ Epoch {epoch} -> Stage {stage} (CoT mode)")
    
    print("  ✓ CoT mode test passed!")


def test_stage_info():
    """Test stage information generation"""
    print("Testing stage information...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 3,
        'c_thought': 2,
        'uniform_prob': 0.1,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Test stage 0 (CoT)
    stage_0 = stage_manager.get_stage_info(0)
    assert stage_0.stage_number == 0
    assert stage_0.is_cot_stage == True
    assert stage_0.num_latent_steps == 0
    assert stage_0.num_latent_tokens == 0
    print(f"  ✓ Stage 0: {stage_0.description}")
    
    # Test stage 1
    stage_1 = stage_manager.get_stage_info(1)
    assert stage_1.stage_number == 1
    assert stage_1.is_cot_stage == False
    assert stage_1.num_latent_steps == 1
    assert stage_1.num_latent_tokens == 2  # 1 * c_thought
    print(f"  ✓ Stage 1: {stage_1.description}")
    
    # Test stage beyond max_latent_stage
    stage_5 = stage_manager.get_stage_info(5)
    assert stage_5.stage_number == 5
    assert stage_5.is_cot_stage == False
    assert stage_5.num_latent_steps == 3  # Capped at max_latent_stage
    assert stage_5.num_latent_tokens == 6  # 3 * c_thought
    print(f"  ✓ Stage 5: {stage_5.description}")
    
    print("  ✓ Stage information test passed!")


def test_effective_stage_calculation():
    """Test effective stage calculation for samples"""
    print("Testing effective stage calculation...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 3,
        'c_thought': 2,
        'uniform_prob': 0.0,  # Disable for predictable testing
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Test normal case
    sample_steps = ["Step 1", "Step 2", "Step 3", "Step 4"]
    effective_stage, n_skip_steps, n_latent_tokens = stage_manager.get_effective_stage_for_sample(2, sample_steps)
    
    assert effective_stage == 2
    assert n_skip_steps == 2
    assert n_latent_tokens == 4  # 2 * c_thought
    print(f"  ✓ Normal case: stage={effective_stage}, skip={n_skip_steps}, latent={n_latent_tokens}")
    
    # Test max_latent_stage constraint
    effective_stage, n_skip_steps, n_latent_tokens = stage_manager.get_effective_stage_for_sample(5, sample_steps)
    
    assert effective_stage == 5
    assert n_skip_steps == 10000  # Skip all (original CoCoNuT pattern)
    assert n_latent_tokens == 6  # min(4, 3) * c_thought = 3 * 2
    print(f"  ✓ Max stage constraint: stage={effective_stage}, skip={n_skip_steps}, latent={n_latent_tokens}")
    
    print("  ✓ Effective stage calculation test passed!")


def test_no_cot_mode():
    """Test no_cot mode"""
    print("Testing no_cot mode...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 3,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': True,  # Enable no_cot mode
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Should always return stage 0 due to no_cot
    stage = stage_manager.get_current_stage(10)
    assert stage == 0, f"no_cot mode should return stage 0, got {stage}"
    
    # Should skip all steps and have 0 latent tokens
    sample_steps = ["Step 1", "Step 2", "Step 3"]
    effective_stage, n_skip_steps, n_latent_tokens = stage_manager.get_effective_stage_for_sample(2, sample_steps)
    
    assert n_skip_steps == 100  # Skip all
    assert n_latent_tokens == 0  # No latent tokens
    print(f"  ✓ no_cot mode: skip={n_skip_steps}, latent={n_latent_tokens}")
    
    print("  ✓ no_cot mode test passed!")


def test_config_updates():
    """Test configuration updates for different stages"""
    print("Testing configuration updates...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 3,
        'c_thought': 2,
        'uniform_prob': 0.1,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Test stage 0 config (should be CoT)
    stage_0_config = stage_manager.update_config_for_stage(0)
    assert stage_0_config.cot == True
    assert stage_0_config.coconut == False
    print("  ✓ Stage 0 config: CoT mode enabled")
    
    # Test stage 1+ config (should be CoCoNuT)
    stage_1_config = stage_manager.update_config_for_stage(1)
    assert stage_1_config.cot == False
    assert stage_1_config.coconut == True
    print("  ✓ Stage 1+ config: CoCoNuT mode enabled")
    
    print("  ✓ Configuration updates test passed!")


def test_curriculum_summary():
    """Test curriculum summary generation"""
    print("Testing curriculum summary...")
    
    config = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.1,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = StageManager(config)
    
    # Generate summary for 10 epochs
    summary = stage_manager.get_training_summary(10)
    
    assert summary['total_epochs'] == 10
    assert summary['epochs_per_stage'] == 3
    assert summary['max_latent_stage'] == 2
    assert summary['c_thought'] == 2
    
    # Should have stages 0, 1, 2, 3
    expected_stages = [0, 1, 2, 3]
    actual_stages = [s['stage'] for s in summary['stages']]
    assert actual_stages == expected_stages, f"Expected stages {expected_stages}, got {actual_stages}"
    
    print(f"  ✓ Summary generated for {len(summary['stages'])} stages")
    
    # Print the summary for visual inspection
    print("\n  Curriculum Summary:")
    stage_manager.print_curriculum_summary(10)
    
    print("  ✓ Curriculum summary test passed!")


def test_factory_function():
    """Test the factory function"""
    print("Testing factory function...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 4,
        'c_thought': 2,
        'uniform_prob': 0.1,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    stage_manager = create_stage_manager(config)
    assert isinstance(stage_manager, StageManager)
    assert stage_manager.epochs_per_stage == 5
    assert stage_manager.max_latent_stage == 4
    
    print("  ✓ Factory function test passed!")


def test_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test invalid epochs_per_stage
    try:
        config = Config({'epochs_per_stage': 0, 'max_latent_stage': 4, 'c_thought': 2, 'uniform_prob': 0.1})
        StageManager(config)
        assert False, "Should have raised ValueError for epochs_per_stage <= 0"
    except ValueError:
        print("  ✓ Correctly rejected epochs_per_stage <= 0")
    
    # Test invalid uniform_prob
    try:
        config = Config({'epochs_per_stage': 5, 'max_latent_stage': 4, 'c_thought': 2, 'uniform_prob': 1.5})
        StageManager(config)
        assert False, "Should have raised ValueError for uniform_prob > 1"
    except ValueError:
        print("  ✓ Correctly rejected uniform_prob > 1")
    
    print("  ✓ Configuration validation test passed!")


def main():
    """Run all tests"""
    print("Running multimodal CoCoNuT stage manager tests...\n")
    
    try:
        test_basic_stage_progression()
        print()
        
        test_cot_mode()
        print()
        
        test_stage_info()
        print()
        
        test_effective_stage_calculation()
        print()
        
        test_no_cot_mode()
        print()
        
        test_config_updates()
        print()
        
        test_curriculum_summary()
        print()
        
        test_factory_function()
        print()
        
        test_validation()
        print()
        
        print("🎉 All stage manager tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())