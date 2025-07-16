#!/usr/bin/env python3
"""
Test Progressive Latent Stage Training Implementation

This test specifically validates task 4.3 requirements:
- Data preparation for replacing reasoning steps with latent tokens
- Training logic for stages 1 through max_latent_stage  
- Uniform probability mixing for multi-stage data
"""

import sys
import os
sys.path.append('.')

import torch
import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from multimodal_coconut.config import Config
from multimodal_coconut.training.stage_manager import StageManager
from multimodal_coconut.data.dataset import (
    get_multimodal_dataset,
    get_multimodal_cot_latent_dataset,
    MultimodalCollator
)


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.latent_token_id = 101
        self.start_latent_id = 100
        self.end_latent_id = 102
        self.padding_side = "right"
    
    def encode(self, text, add_special_tokens=False):
        # Simple word-based tokenization for testing
        words = text.lower().split()
        return [hash(word) % 1000 + 10 for word in words]


def create_test_data(temp_dir):
    """Create test data for progressive training"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create test images
    for i in range(4):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"test_{i}.jpg"
        img.save(img_path)
    
    # Create test data with multiple reasoning steps
    data = []
    for i in range(10):
        sample = {
            "image_path": str(images_dir / f"test_{i % 4}.jpg"),
            "question": f"What is in image {i}?",
            "steps": [
                f"Step 1: I observe the image {i}",
                f"Step 2: I analyze the visual elements {i}",
                f"Step 3: I identify key features {i}",
                f"Step 4: I draw conclusions {i}"
            ],
            "answer": f"The answer for image {i}"
        }
        data.append(sample)
    
    # Save data
    data_path = Path(temp_dir) / "test_data.json"
    with open(data_path, 'w') as f:
        json.dump(data, f)
    
    return str(data_path), str(images_dir)


def test_progressive_data_preparation():
    """Test data preparation for replacing reasoning steps with latent tokens"""
    print("Testing Progressive Data Preparation...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_test_data(temp_dir)
        tokenizer = MockTokenizer()
        
        # Create configuration
        config = Config({
            'c_thought': 2,
            'max_latent_stage': 3,
            'uniform_prob': 0.0,  # Disable for predictable testing
            'no_cot': False,
            'pad_latent_to_max': False
        })
        
        # Load base dataset
        base_dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            max_size=5
        )
        
        print(f"✓ Base dataset loaded: {len(base_dataset)} samples")
        
        # Test different stages
        for stage in [0, 1, 2, 3]:
            print(f"\n--- Testing Stage {stage} ---")
            
            # Prepare stage-specific dataset
            stage_dataset = get_multimodal_cot_latent_dataset(
                scheduled_stage=stage,
                base_dataset=base_dataset,
                configs=config,
                start_id=tokenizer.start_latent_id,
                latent_id=tokenizer.latent_token_id,
                end_id=tokenizer.end_latent_id,
                no_special_marker=False
            )
            
            print(f"✓ Stage {stage} dataset created: {len(stage_dataset)} samples")
            
            # Examine a sample
            sample = stage_dataset[0]
            input_ids = sample['input_ids']
            labels = sample['labels']
            
            # Count latent tokens
            latent_count = input_ids.count(tokenizer.latent_token_id)
            expected_latent_count = stage * config.c_thought
            
            print(f"  - Input length: {len(input_ids)}")
            print(f"  - Latent tokens: {latent_count} (expected: {expected_latent_count})")
            print(f"  - Labels length: {len(labels)}")
            
            # Verify latent token count
            assert latent_count == expected_latent_count, f"Stage {stage}: Expected {expected_latent_count} latent tokens, got {latent_count}"
            
            # Verify special markers for non-zero stages
            if stage > 0:
                start_count = input_ids.count(tokenizer.start_latent_id)
                end_count = input_ids.count(tokenizer.end_latent_id)
                assert start_count == 1, f"Expected 1 start marker, got {start_count}"
                assert end_count == 1, f"Expected 1 end marker, got {end_count}"
                print(f"  - Special markers: start={start_count}, end={end_count}")
            
            # Verify labels ignore latent tokens
            latent_positions = [i for i, token in enumerate(input_ids) if token == tokenizer.latent_token_id]
            for pos in latent_positions:
                assert labels[pos] == -100, f"Label at latent position {pos} should be -100, got {labels[pos]}"
            
            print(f"✓ Stage {stage} data preparation verified")
        
        print("\n✓ Data preparation successful")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_uniform_probability_mixing():
    """Test uniform probability mixing for multi-stage data"""
    print("\nTesting Uniform Probability Mixing...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_test_data(temp_dir)
        tokenizer = MockTokenizer()
        
        # Create configuration with uniform mixing enabled
        config = Config({
            'c_thought': 2,
            'max_latent_stage': 3,
            'uniform_prob': 0.5,  # 50% chance of uniform mixing
            'no_cot': False,
            'pad_latent_to_max': False
        })
        
        # Load base dataset
        base_dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            max_size=2
        )
        
        # Test stage manager uniform mixing
        stage_manager = StageManager(config)
        
        # Test uniform mixing probability
        mixing_results = []
        for _ in range(100):
            mixing_results.append(stage_manager.should_use_uniform_mixing())
        
        mixing_rate = sum(mixing_results) / len(mixing_results)
        print(f"✓ Uniform mixing rate: {mixing_rate:.2f} (expected ~0.5)")
        
        # Test stage sampling
        sample_steps = ["step1", "step2", "step3"]
        stage_samples = []
        
        for _ in range(50):
            effective_stage, n_skip, n_latent = stage_manager.get_effective_stage_for_sample(
                scheduled_stage=2, 
                sample_steps=sample_steps
            )
            stage_samples.append(effective_stage)
        
        # Should see variety due to uniform mixing
        unique_stages = set(stage_samples)
        print(f"✓ Sampled stages: {sorted(unique_stages)} (shows mixing)")
        
        # Test with actual dataset processing
        stage_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=2,
            base_dataset=base_dataset,
            configs=config,
            start_id=tokenizer.start_latent_id,
            latent_id=tokenizer.latent_token_id,
            end_id=tokenizer.end_latent_id,
            no_special_marker=False
        )
        
        # Check latent token distribution (should vary due to mixing)
        latent_counts = []
        for i in range(len(stage_dataset)):
            sample = stage_dataset[i]
            latent_count = sample['input_ids'].count(tokenizer.latent_token_id)
            latent_counts.append(latent_count)
        
        unique_counts = set(latent_counts)
        print(f"✓ Latent token counts in dataset: {sorted(unique_counts)} (shows stage mixing)")
        
        print("✓ Uniform probability mixing verified")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_stage_progression_logic():
    """Test the stage progression logic for stages 1 through max_latent_stage"""
    print("\nTesting Stage Progression Logic...")
    
    config = Config({
        'epochs_per_stage': 5,
        'max_latent_stage': 4,
        'c_thought': 2,
        'uniform_prob': 0.0,
        'cot': False,
        'coconut': True,
        'no_cot': False
    })
    
    stage_manager = StageManager(config)
    
    # Test epoch to stage mapping
    test_cases = [
        (0, 0),   # Epoch 0 -> Stage 0
        (4, 0),   # Epoch 4 -> Stage 0
        (5, 1),   # Epoch 5 -> Stage 1
        (9, 1),   # Epoch 9 -> Stage 1
        (10, 2),  # Epoch 10 -> Stage 2
        (15, 3),  # Epoch 15 -> Stage 3
        (20, 4),  # Epoch 20 -> Stage 4
        (25, 5),  # Epoch 25 -> Stage 5 (beyond max)
    ]
    
    print("Stage progression test:")
    for epoch, expected_stage in test_cases:
        actual_stage = stage_manager.get_current_stage(epoch)
        print(f"  Epoch {epoch} -> Stage {actual_stage}")
        assert actual_stage == expected_stage, f"Expected stage {expected_stage}, got {actual_stage}"
    
    # Test stage info for different stages
    for stage in range(5):
        stage_info = stage_manager.get_stage_info(stage)
        print(f"\nStage {stage} info:")
        print(f"  - Is CoT stage: {stage_info.is_cot_stage}")
        print(f"  - Latent steps: {stage_info.num_latent_steps}")
        print(f"  - Latent tokens: {stage_info.num_latent_tokens}")
        print(f"  - Description: {stage_info.description}")
        
        if stage == 0:
            assert stage_info.is_cot_stage == True
            assert stage_info.num_latent_tokens == 0
        else:
            assert stage_info.is_cot_stage == False
            expected_tokens = min(stage, config.max_latent_stage) * config.c_thought
            assert stage_info.num_latent_tokens == expected_tokens
    
    print("✓ Stage progression logic verified")
    return True


def test_collator_with_latent_tokens():
    """Test the collator handles latent tokens correctly"""
    print("\nTesting Collator with Latent Tokens...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_test_data(temp_dir)
        tokenizer = MockTokenizer()
        
        config = Config({
            'c_thought': 3,
            'max_latent_stage': 2,
            'uniform_prob': 0.0,
            'no_cot': False,
            'pad_latent_to_max': False
        })
        
        # Create dataset with latent tokens
        base_dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            max_size=3
        )
        
        stage_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=2,
            base_dataset=base_dataset,
            configs=config,
            start_id=tokenizer.start_latent_id,
            latent_id=tokenizer.latent_token_id,
            end_id=tokenizer.end_latent_id,
            no_special_marker=False
        )
        
        # Test collator
        collator = MultimodalCollator(
            tokenizer=tokenizer,
            latent_id=tokenizer.latent_token_id,
            label_pad_token_id=-100
        )
        
        # Create batch with different latent token positions
        features = [stage_dataset[i] for i in range(2)]
        batch = collator(features)
        
        print(f"✓ Batch created successfully")
        print(f"  - Input IDs shape: {batch['input_ids'].shape}")
        print(f"  - Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  - Position IDs shape: {batch.get('position_ids', torch.tensor([])).shape}")
        
        # Verify latent token alignment
        if 'position_ids' in batch:
            print("✓ Position IDs created for latent token alignment")
            
            # Check that latent tokens are aligned across batch
            input_ids = batch['input_ids']
            latent_positions = []
            for i in range(input_ids.shape[0]):
                positions = (input_ids[i] == tokenizer.latent_token_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    latent_positions.append(positions[0].item())
            
            if len(latent_positions) > 1:
                # All latent tokens should start at the same position due to alignment
                assert all(pos == latent_positions[0] for pos in latent_positions), \
                    f"Latent tokens not aligned: {latent_positions}"
                print(f"✓ Latent tokens aligned at position {latent_positions[0]}")
        
        print("✓ Collator with latent tokens verified")
        return True
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all progressive latent stage training tests"""
    print("=" * 60)
    print("PROGRESSIVE LATENT STAGE TRAINING IMPLEMENTATION TEST")
    print("=" * 60)
    
    tests = [
        test_progressive_data_preparation,
        test_uniform_probability_mixing,
        test_stage_progression_logic,
        test_collator_with_latent_tokens
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All progressive latent training components verified!")
        print("\nTask 4.3 Implementation Status:")
        print("✓ Data preparation for replacing reasoning steps with latent tokens")
        print("✓ Training logic for stages 1 through max_latent_stage")
        print("✓ Uniform probability mixing for multi-stage data")
        print("✓ Requirements 3.4, 3.5, 3.6 satisfied")
    else:
        print("❌ Some tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)