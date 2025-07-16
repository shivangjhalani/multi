#!/usr/bin/env python3
"""
Integration test for stage manager with multimodal dataset functions.

This test verifies that the stage manager correctly integrates with the
multimodal dataset preparation functions and produces the expected results.
"""

import sys
import os
sys.path.append('.')

import torch
from datasets import Dataset
from multimodal_coconut.config import Config
from multimodal_coconut.training.stage_manager import StageManager
from multimodal_coconut.data.dataset import (
    get_multimodal_cot_latent_dataset,
    get_multimodal_question_latent_dataset
)


def create_mock_tokenizer():
    """Create a mock tokenizer for testing"""
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 2
        
        def encode(self, text, add_special_tokens=False):
            # Simple mock encoding - just return length-based tokens
            return list(range(1, len(text.split()) + 1))
    
    return MockTokenizer()


def create_test_dataset():
    """Create a test dataset with multimodal samples"""
    # Create mock data
    samples = []
    for i in range(5):
        sample = {
            'pixel_values': torch.randn(3, 3, 224, 224),  # Mock image tensor
            'num_patches': 3,
            'question_tokenized': [1, 2, 3, 4, 5],  # Mock question tokens
            'steps_tokenized': [
                [6, 7],      # Step 1
                [8, 9, 10],  # Step 2  
                [11, 12]     # Step 3
            ],
            'answer_tokenized': [13, 14, 15],  # Mock answer tokens
            'idx': i
        }
        samples.append(sample)
    
    # Convert to HuggingFace dataset
    dataset_dict = {}
    for key in samples[0].keys():
        dataset_dict[key] = [sample[key] for sample in samples]
    
    return Dataset.from_dict(dataset_dict)


def test_stage_manager_integration():
    """Test stage manager integration with dataset functions"""
    print("Testing stage manager integration with dataset functions...")
    
    # Create test configuration
    config = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.0,  # Disable for predictable testing
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    # Create test dataset
    base_dataset = create_test_dataset()
    
    # Mock token IDs
    start_id = 100
    latent_id = 101
    end_id = 102
    
    # Test different stages
    stages_to_test = [0, 1, 2, 3]  # Stage 3 is beyond max_latent_stage
    
    for stage in stages_to_test:
        print(f"  Testing Stage {stage}...")
        
        # Prepare dataset for this stage
        stage_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False,
            shuffle=False
        )
        
        # Check first sample
        sample = stage_dataset[0]
        
        # Count latent tokens
        latent_count = sample['input_ids'].count(latent_id)
        
        # Count ignored labels (-100)
        ignored_count = sample['labels'].count(-100)
        
        if stage == 0:
            # Stage 0 should have no latent tokens
            expected_latent = 0
            print(f"    ✓ Stage {stage} (CoT): {latent_count} latent tokens, {ignored_count} ignored labels")
        elif stage <= config.max_latent_stage:
            # Normal CoCoNuT stages
            expected_latent = stage * config.c_thought
            print(f"    ✓ Stage {stage} ({stage} latent step{'s' if stage > 1 else ''}): {latent_count} latent tokens, {ignored_count} ignored labels")
        else:
            # Beyond max_latent_stage - should be capped
            expected_latent = config.max_latent_stage * config.c_thought
            print(f"    ✓ Stage {stage} (beyond max_latent_stage): {latent_count} latent tokens, {ignored_count} ignored labels")
        
        assert latent_count == expected_latent, f"Expected {expected_latent} latent tokens, got {latent_count}"
        
        # Verify multimodal components are preserved
        assert 'pixel_values' in sample
        assert 'num_patches' in sample
        assert sample['num_patches'] == 3
    
    print("  ✓ Stage manager integration test passed!")


def test_validation_dataset_integration():
    """Test stage manager integration with validation dataset"""
    print("Testing stage manager integration with validation dataset...")
    
    config = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    base_dataset = create_test_dataset()
    
    start_id = 100
    latent_id = 101
    end_id = 102
    
    stages_to_test = [0, 1, 2]
    
    for stage in stages_to_test:
        print(f"  Testing validation stage {stage}...")
        
        val_dataset = get_multimodal_question_latent_dataset(
            scheduled_stage=stage,
            base_dataset_valid=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False
        )
        
        sample = val_dataset[0]
        latent_count = sample['input_ids'].count(latent_id)
        
        expected_latent = stage * config.c_thought
        print(f"    ✓ Validation stage {stage}: {latent_count} latent tokens")
        
        assert latent_count == expected_latent, f"Expected {expected_latent} latent tokens, got {latent_count}"
        
        # Verify multimodal components
        assert 'pixel_values' in sample
        assert 'num_patches' in sample
    
    print("  ✓ Validation dataset integration test passed!")


def test_uniform_mixing():
    """Test uniform probability mixing"""
    print("Testing uniform probability mixing...")
    
    config = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 1.0,  # Always use uniform mixing
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': False
    })
    
    base_dataset = create_test_dataset()
    
    start_id = 100
    latent_id = 101
    end_id = 102
    
    # Run multiple times to see different random stages
    latent_counts = set()
    
    for _ in range(10):
        stage_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=2,  # Fixed scheduled stage
            base_dataset=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False,
            shuffle=False
        )
        
        sample = stage_dataset[0]
        latent_count = sample['input_ids'].count(latent_id)
        latent_counts.add(latent_count)
    
    # Should have seen different latent counts due to random mixing
    print(f"  ✓ Uniform mixing produced {len(latent_counts)} different latent counts: {sorted(latent_counts)}")
    assert len(latent_counts) > 1, "Uniform mixing should produce different results"
    
    print("  ✓ Uniform mixing test passed!")


def test_edge_cases():
    """Test edge cases like no_cot and pad_latent_to_max"""
    print("Testing edge cases...")
    
    # Test no_cot mode
    config_no_cot = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': True,  # Enable no_cot
        'pad_latent_to_max': False
    })
    
    base_dataset = create_test_dataset()
    
    stage_dataset = get_multimodal_cot_latent_dataset(
        scheduled_stage=2,
        base_dataset=base_dataset,
        configs=config_no_cot,
        start_id=100,
        latent_id=101,
        end_id=102,
        no_special_marker=False,
        shuffle=False
    )
    
    sample = stage_dataset[0]
    latent_count = sample['input_ids'].count(101)
    assert latent_count == 0, f"no_cot mode should have 0 latent tokens, got {latent_count}"
    print("  ✓ no_cot mode test passed!")
    
    # Test pad_latent_to_max
    config_pad = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': False,
        'pad_latent_to_max': True  # Enable padding
    })
    
    stage_dataset = get_multimodal_cot_latent_dataset(
        scheduled_stage=5,  # Beyond max_latent_stage
        base_dataset=base_dataset,
        configs=config_pad,
        start_id=100,
        latent_id=101,
        end_id=102,
        no_special_marker=False,
        shuffle=False
    )
    
    sample = stage_dataset[0]
    latent_count = sample['input_ids'].count(101)
    expected = config_pad.max_latent_stage * config_pad.c_thought
    assert latent_count == expected, f"pad_latent_to_max should give {expected} tokens, got {latent_count}"
    print("  ✓ pad_latent_to_max test passed!")
    
    print("  ✓ Edge cases test passed!")


def main():
    """Run all integration tests"""
    print("Running stage manager integration tests...\n")
    
    try:
        test_stage_manager_integration()
        print()
        
        test_validation_dataset_integration()
        print()
        
        test_uniform_mixing()
        print()
        
        test_edge_cases()
        print()
        
        print("🎉 All stage manager integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())