#!/usr/bin/env python3
"""
Test script for stage manager integration with multimodal dataset functions.

This script validates that the stage manager correctly integrates with the
multimodal dataset processing functions and produces the expected results.
"""

import sys
import os
sys.path.append('.')

import torch
from datasets import Dataset
from multimodal_coconut.config import Config
from multimodal_coconut.training.stage_manager import StageManager


def create_mock_multimodal_dataset():
    """Create a mock multimodal dataset for testing"""
    # Create mock tokenized samples
    samples = []
    for i in range(5):
        sample = {
            'pixel_values': torch.randn(2, 3, 224, 224),  # 2 patches per image
            'num_patches': 2,
            'question_tokenized': [1, 2, 3, 4, 5],  # Mock question tokens
            'steps_tokenized': [
                [10, 11, 12],  # Step 1
                [20, 21, 22],  # Step 2
                [30, 31, 32],  # Step 3
            ],
            'answer_tokenized': [100, 101, 102],  # Mock answer tokens
            'idx': i
        }
        samples.append(sample)
    
    # Convert to HuggingFace dataset format
    dataset_dict = {}
    for key in samples[0].keys():
        if key == 'pixel_values':
            # Convert tensors to lists for HF dataset compatibility
            dataset_dict[key] = [sample[key].tolist() for sample in samples]
        else:
            dataset_dict[key] = [sample[key] for sample in samples]
    
    return Dataset.from_dict(dataset_dict)


def test_stage_manager_integration():
    """Test that stage manager integrates correctly with dataset functions"""
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
    
    # Create mock dataset
    base_dataset = create_mock_multimodal_dataset()
    
    # Import the dataset function
    from multimodal_coconut.data.dataset import get_multimodal_cot_latent_dataset
    
    # Test different stages
    test_cases = [
        (0, "Stage 0 (CoT)"),
        (1, "Stage 1 (1 latent step)"),
        (2, "Stage 2 (2 latent steps)"),
        (3, "Stage 3 (beyond max_latent_stage)"),
    ]
    
    # Mock token IDs
    start_id = 1000
    latent_id = 1001
    end_id = 1002
    
    for scheduled_stage, description in test_cases:
        print(f"\n  Testing {description}...")
        
        try:
            # Process dataset for this stage
            processed_dataset = get_multimodal_cot_latent_dataset(
                scheduled_stage=scheduled_stage,
                base_dataset=base_dataset,
                configs=config,
                start_id=start_id,
                latent_id=latent_id,
                end_id=end_id,
                no_special_marker=False,
                shuffle=False
            )
            
            # Verify the processed dataset
            assert len(processed_dataset) == len(base_dataset), f"Dataset size mismatch for {description}"
            
            # Check first sample
            sample = processed_dataset[0]
            
            # Verify required fields
            required_fields = ['pixel_values', 'num_patches', 'input_ids', 'labels', 'attention_mask', 'idx', 'position_ids']
            for field in required_fields:
                assert field in sample, f"Missing field {field} in {description}"
            
            # Verify pixel values are preserved
            assert isinstance(sample['pixel_values'], list), f"pixel_values should be list in {description}"
            assert len(sample['pixel_values']) == 2, f"Should have 2 patches in {description}"
            
            # Verify num_patches
            assert sample['num_patches'] == 2, f"num_patches mismatch in {description}"
            
            # Verify input_ids structure
            input_ids = sample['input_ids']
            assert isinstance(input_ids, list), f"input_ids should be list in {description}"
            
            # Count latent tokens in input_ids
            latent_count = input_ids.count(latent_id)
            
            # Calculate expected latent tokens based on stage
            stage_manager = StageManager(config)
            effective_stage, n_skip_steps, n_latent_tokens = stage_manager.get_effective_stage_for_sample(
                scheduled_stage, [[], [], []]  # 3 steps
            )
            
            assert latent_count == n_latent_tokens, f"Expected {n_latent_tokens} latent tokens, got {latent_count} in {description}"
            
            # Verify labels structure
            labels = sample['labels']
            assert len(labels) == len(input_ids), f"Labels length mismatch in {description}"
            
            # Count -100 labels (should cover question + latent tokens + markers)
            ignore_count = labels.count(-100)
            expected_ignore = len([1, 2, 3, 4, 5]) + n_latent_tokens + 2  # question + latent + start/end markers
            assert ignore_count == expected_ignore, f"Expected {expected_ignore} ignored labels, got {ignore_count} in {description}"
            
            print(f"    ✓ {description}: {latent_count} latent tokens, {ignore_count} ignored labels")
            
        except Exception as e:
            print(f"    ❌ {description} failed: {e}")
            raise
    
    print("  ✓ Stage manager integration test passed!")


def test_validation_dataset_integration():
    """Test stage manager integration with validation dataset function"""
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
    
    # Create mock validation dataset
    base_dataset = create_mock_multimodal_dataset()
    
    # Import the validation dataset function
    from multimodal_coconut.data.dataset import get_multimodal_question_latent_dataset
    
    # Mock token IDs
    start_id = 1000
    latent_id = 1001
    end_id = 1002
    
    # Test different stages
    for scheduled_stage in [0, 1, 2]:
        print(f"  Testing validation stage {scheduled_stage}...")
        
        try:
            # Process validation dataset
            processed_dataset = get_multimodal_question_latent_dataset(
                scheduled_stage=scheduled_stage,
                base_dataset_valid=base_dataset,
                configs=config,
                start_id=start_id,
                latent_id=latent_id,
                end_id=end_id,
                no_special_marker=False
            )
            
            # Verify the processed dataset
            assert len(processed_dataset) == len(base_dataset)
            
            # Check first sample
            sample = processed_dataset[0]
            
            # Verify required fields for validation
            required_fields = ['pixel_values', 'num_patches', 'input_ids', 'idx', 'attention_mask', 'position_ids']
            for field in required_fields:
                assert field in sample, f"Missing field {field} in validation stage {scheduled_stage}"
            
            # Verify no labels in validation dataset
            assert 'labels' not in sample, f"Validation dataset should not have labels in stage {scheduled_stage}"
            
            # Count latent tokens
            input_ids = sample['input_ids']
            latent_count = input_ids.count(latent_id)
            
            # Expected latent tokens for validation
            expected_latent = min(config.max_latent_stage, scheduled_stage) * config.c_thought
            assert latent_count == expected_latent, f"Expected {expected_latent} latent tokens, got {latent_count} in validation stage {scheduled_stage}"
            
            print(f"    ✓ Validation stage {scheduled_stage}: {latent_count} latent tokens")
            
        except Exception as e:
            print(f"    ❌ Validation stage {scheduled_stage} failed: {e}")
            raise
    
    print("  ✓ Validation dataset integration test passed!")


def test_uniform_mixing():
    """Test uniform probability mixing functionality"""
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
    
    # Create mock dataset
    base_dataset = create_mock_multimodal_dataset()
    
    from multimodal_coconut.data.dataset import get_multimodal_cot_latent_dataset
    
    # Mock token IDs
    start_id = 1000
    latent_id = 1001
    end_id = 1002
    
    # Process dataset multiple times to see variation
    latent_counts = []
    
    for i in range(10):
        processed_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=2,  # Fixed stage
            base_dataset=base_dataset,
            configs=config,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False,
            shuffle=False
        )
        
        # Count latent tokens in first sample
        sample = processed_dataset[0]
        latent_count = sample['input_ids'].count(latent_id)
        latent_counts.append(latent_count)
    
    # Should see variation due to uniform mixing
    unique_counts = set(latent_counts)
    assert len(unique_counts) > 1, f"Expected variation in latent counts due to uniform mixing, got {unique_counts}"
    
    print(f"  ✓ Uniform mixing produced {len(unique_counts)} different latent counts: {sorted(unique_counts)}")
    print("  ✓ Uniform mixing test passed!")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases...")
    
    # Test with no_cot mode
    config = Config({
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': True,  # Enable no_cot mode
        'pad_latent_to_max': False
    })
    
    base_dataset = create_mock_multimodal_dataset()
    
    from multimodal_coconut.data.dataset import get_multimodal_cot_latent_dataset
    
    # Mock token IDs
    start_id = 1000
    latent_id = 1001
    end_id = 1002
    
    processed_dataset = get_multimodal_cot_latent_dataset(
        scheduled_stage=2,
        base_dataset=base_dataset,
        configs=config,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
        shuffle=False
    )
    
    # Should have no latent tokens in no_cot mode
    sample = processed_dataset[0]
    latent_count = sample['input_ids'].count(latent_id)
    assert latent_count == 0, f"Expected 0 latent tokens in no_cot mode, got {latent_count}"
    
    print("  ✓ no_cot mode test passed!")
    
    # Test with pad_latent_to_max
    config.no_cot = False
    config.pad_latent_to_max = True
    
    processed_dataset = get_multimodal_cot_latent_dataset(
        scheduled_stage=5,  # Beyond max_latent_stage
        base_dataset=base_dataset,
        configs=config,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
        shuffle=False
    )
    
    # Should be capped at max_latent_stage
    sample = processed_dataset[0]
    latent_count = sample['input_ids'].count(latent_id)
    expected_count = config.max_latent_stage * config.c_thought
    assert latent_count == expected_count, f"Expected {expected_count} latent tokens with pad_latent_to_max, got {latent_count}"
    
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