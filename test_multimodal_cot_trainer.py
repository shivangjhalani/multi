#!/usr/bin/env python3
"""
Test script for the multimodal CoT trainer.

This script validates that the multimodal CoT trainer correctly implements
Stage 0 training (standard multimodal chain-of-thought) and integrates
properly with the multimodal data pipeline and model.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from datasets import Dataset
from multimodal_coconut.config import Config
from multimodal_coconut.training.multimodal_cot_trainer import MultimodalCoTTrainer, create_multimodal_cot_trainer


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
        # Simple mock encoding
        return list(range(1, len(text.split()) + 1))
    
    def decode(self, token_id):
        return f"token_{token_id}"


class MockModel(nn.Module):
    """Mock multimodal model for testing"""
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids, labels=None, pixel_values=None, **kwargs):
        # Simple forward pass for testing
        batch_size, seq_len = input_ids.shape
        
        # Mock embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Mock logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Mock output object
        class MockOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return MockOutput(loss, logits)


def create_mock_dataset():
    """Create a mock multimodal dataset for testing"""
    samples = []
    for i in range(10):
        sample = {
            'pixel_values': torch.randn(2, 3, 224, 224),  # Mock image tensor
            'num_patches': 2,
            'input_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Mock tokens
            'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],     # Mock labels
            'attention_mask': [1] * 10,
            'idx': i
        }
        samples.append(sample)
    
    # Convert to HuggingFace dataset
    dataset_dict = {}
    for key in samples[0].keys():
        dataset_dict[key] = [sample[key] for sample in samples]
    
    return Dataset.from_dict(dataset_dict)


def test_trainer_initialization():
    """Test trainer initialization"""
    print("Testing trainer initialization...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'cot': True,
        'coconut': False,
        'c_thought': 2,
        'max_latent_stage': 3,
        'epochs_per_stage': 5
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Check initialization
    assert trainer.model is model
    assert trainer.tokenizer is tokenizer
    assert trainer.config is config
    assert trainer.rank == 0
    assert trainer.world_size == 1
    assert trainer.current_epoch == 0
    assert trainer.total_train_steps == 0
    
    # Check CoT configuration validation
    assert trainer.config.cot == True
    assert trainer.config.coconut == False
    
    print("  ✓ Trainer initialization test passed!")


def test_config_validation():
    """Test configuration validation for CoT training"""
    print("Testing configuration validation...")
    
    # Test with coconut=True (should be corrected)
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'cot': False,  # Wrong setting
        'coconut': True,  # Wrong setting
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Should be corrected during initialization
    assert trainer.config.cot == True
    assert trainer.config.coconut == False
    
    print("  ✓ Configuration validation test passed!")


def test_optimizer_setup():
    """Test optimizer setup"""
    print("Testing optimizer setup...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 2,
        'cot': True,
        'coconut': False
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    optimizer = trainer.setup_optimizer()
    
    # Check optimizer type and parameters
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert optimizer.param_groups[0]['weight_decay'] == 0.01
    
    print("  ✓ Optimizer setup test passed!")


def test_learning_rate_scaling():
    """Test learning rate scaling for distributed training"""
    print("Testing learning rate scaling...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'num_epochs': 2,
        'cot': True,
        'coconut': False
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=2  # Simulate 2 processes
    )
    
    optimizer = trainer.setup_optimizer()
    
    # Check that learning rate was scaled
    # effective_batch_size = 2 * 2 * 4 = 16
    # base_batch_size = 2 * 2 = 4
    # scaling_factor = 16 / 4 = 4
    # scaled_lr = 1e-4 * 4 = 4e-4
    expected_lr = 1e-4 * 4
    actual_lr = optimizer.param_groups[0]['lr']
    
    assert abs(actual_lr - expected_lr) < 1e-6, f"Expected LR {expected_lr}, got {actual_lr}"
    
    print(f"  ✓ Learning rate scaling test passed! (Base: 1e-4, Scaled: {actual_lr})")


def test_stage_manager_integration():
    """Test integration with stage manager"""
    print("Testing stage manager integration...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        'cot': True,
        'coconut': False
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Check stage manager
    assert trainer.stage_manager is not None
    assert trainer.stage_manager.epochs_per_stage == 3
    assert trainer.stage_manager.max_latent_stage == 2
    assert trainer.stage_manager.c_thought == 2
    
    # For CoT training, should always return stage 0
    for epoch in [0, 5, 10, 15]:
        stage = trainer.stage_manager.get_current_stage(epoch)
        assert stage == 0, f"CoT training should always be stage 0, got {stage} for epoch {epoch}"
    
    print("  ✓ Stage manager integration test passed!")


def test_factory_function():
    """Test the factory function"""
    print("Testing factory function...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 2,
        'cot': True,
        'coconut': False
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = create_multimodal_cot_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    assert isinstance(trainer, MultimodalCoTTrainer)
    assert trainer.model is model
    assert trainer.tokenizer is tokenizer
    assert trainer.config is config
    
    print("  ✓ Factory function test passed!")


def test_checkpoint_saving():
    """Test checkpoint saving functionality"""
    print("Testing checkpoint saving...")
    
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = Config({
            'name': 'test_cot',
            'save_path': temp_dir,
            'batch_size_training': 2,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'cot': True,
            'coconut': False
        })
        
        model = MockModel()
        tokenizer = MockTokenizer()
        
        trainer = MultimodalCoTTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=0,
            world_size=1
        )
        
        # Test checkpoint saving
        metrics = {'val_loss': 0.5, 'train_loss': 0.6}
        trainer.save_checkpoint(epoch=0, metrics=metrics)
        
        # Check that checkpoint was saved
        checkpoint_path = trainer.save_dir / "checkpoint_1"
        assert checkpoint_path.exists(), f"Checkpoint not saved at {checkpoint_path}"
        
        # Check that best model was saved (first save is always best)
        best_path = trainer.save_dir / "best_cot_model"
        assert best_path.exists(), f"Best model not saved at {best_path}"
        
        # Test that better validation loss updates best model
        better_metrics = {'val_loss': 0.3, 'train_loss': 0.4}
        trainer.save_checkpoint(epoch=1, metrics=better_metrics)
        
        assert trainer.best_val_loss == 0.3
        
        print("  ✓ Checkpoint saving test passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_training_step_simulation():
    """Test a simulated training step"""
    print("Testing training step simulation...")
    
    config = Config({
        'name': 'test_cot',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 1,
        'gradient_accumulation_steps': 1,
        'cot': True,
        'coconut': False
    })
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Create mock batch
    batch = {
        'input_ids': torch.randint(1, 100, (2, 10)),
        'labels': torch.randint(1, 100, (2, 10)),
        'pixel_values': torch.randn(4, 3, 224, 224),  # 2 samples * 2 patches each
        'attention_mask': torch.ones(2, 10),
        'num_patches_list': [2, 2],
        'image_flags': torch.ones(2, 1)
    }
    
    # Test forward pass
    model.train()
    outputs = model(**batch)
    
    assert outputs.loss is not None
    assert outputs.logits is not None
    assert outputs.logits.shape == (2, 10, 1000)  # batch_size, seq_len, vocab_size
    
    # Test that loss is reasonable
    assert outputs.loss.item() > 0
    assert not torch.isnan(outputs.loss)
    
    print("  ✓ Training step simulation test passed!")


def main():
    """Run all tests"""
    print("Running multimodal CoT trainer tests...\n")
    
    try:
        test_trainer_initialization()
        print()
        
        test_config_validation()
        print()
        
        test_optimizer_setup()
        print()
        
        test_learning_rate_scaling()
        print()
        
        test_stage_manager_integration()
        print()
        
        test_factory_function()
        print()
        
        test_checkpoint_saving()
        print()
        
        test_training_step_simulation()
        print()
        
        print("🎉 All multimodal CoT trainer tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())