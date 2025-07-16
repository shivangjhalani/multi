#!/usr/bin/env python3
"""
Key Integration Test for Multimodal CoCoNuT

This test focuses on the most critical integration points to ensure
the core functionality works correctly:

1. Stage Manager + Dataset Integration
2. Model + Trainer Integration  
3. Complete Training Pipeline
4. Multimodal Data Flow

This is a focused test to quickly validate the implementation.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import tempfile
import shutil
import json
from pathlib import Path
from PIL import Image
import numpy as np

from multimodal_coconut.config import Config
from multimodal_coconut.data.dataset import get_multimodal_dataset, get_multimodal_cot_latent_dataset
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from multimodal_coconut.training import StageManager, create_multimodal_cot_trainer


class SimpleInternVL3Mock(nn.Module):
    """Simplified mock for focused testing"""
    def __init__(self):
        super().__init__()
        self.language_model = SimpleLMMock()
        self.img_context_token_id = 32000
        
        class Config:
            use_return_dict = True
        self.config = Config()
    
    def extract_feature(self, pixel_values):
        return torch.randn(pixel_values.shape[0], 64)
    
    def forward(self, **kwargs):
        return self.language_model(**kwargs)


class SimpleLMMock(nn.Module):
    """Simplified language model mock"""
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(1000, 64)
        self.lm_head = nn.Linear(64, 1000)
        
        class Config:
            hidden_size = 64
        self.config = Config()
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            hidden = self.embeddings(input_ids)
        
        logits = self.lm_head(hidden)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
                self.hidden_states = [hidden]
                self.past_key_values = None
                self.attentions = None
        
        return Output(loss, logits)


class SimpleTokenizer:
    """Simplified tokenizer for focused testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.latent_token_id = 101
        self.start_latent_id = 100
        self.end_latent_id = 102
        self.padding_side = "right"
    
    def encode(self, text, add_special_tokens=False):
        # Simple word-count based encoding
        return list(range(1, len(text.split()) + 1))
    
    def decode(self, token_id):
        return f"token_{token_id}"


def create_simple_test_data(temp_dir):
    """Create minimal test data"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir()
    
    # Create 3 test images
    for i in range(3):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(images_dir / f"img_{i}.jpg")
    
    # Create test data
    data = []
    for i in range(3):
        data.append({
            "image_path": str(images_dir / f"img_{i}.jpg"),
            "question": f"What is in image {i}?",
            "steps": [f"Step 1 for image {i}", f"Step 2 for image {i}"],
            "answer": f"Answer for image {i}"
        })
    
    # Save data
    data_path = Path(temp_dir) / "data.json"
    with open(data_path, 'w') as f:
        json.dump(data, f)
    
    return str(data_path), str(images_dir)


def test_stage_dataset_integration():
    """Test Stage Manager + Dataset Integration"""
    print("🔗 Testing Stage Manager + Dataset Integration...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_simple_test_data(temp_dir)
        tokenizer = SimpleTokenizer()
        
        # Create config and stage manager
        config = Config({
            'c_thought': 2,
            'max_latent_stage': 2,
            'epochs_per_stage': 3,
            'uniform_prob': 0.0,
            'cot': False,
            'coconut': True,
            'no_cot': False,
            'pad_latent_to_max': False
        })
        
        stage_manager = StageManager(config)
        
        # Load base dataset
        base_dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224
        )
        
        # Test different stages
        for stage in [0, 1, 2]:
            print(f"  Testing stage {stage}...")
            
            # Get stage info
            stage_info = stage_manager.get_stage_info(stage)
            
            # Prepare dataset for this stage
            stage_dataset = get_multimodal_cot_latent_dataset(
                scheduled_stage=stage,
                base_dataset=base_dataset,
                configs=config,
                start_id=tokenizer.start_latent_id,
                latent_id=tokenizer.latent_token_id,
                end_id=tokenizer.end_latent_id,
                no_special_marker=False
            )
            
            # Verify dataset
            sample = stage_dataset[0]
            latent_count = sample['input_ids'].count(tokenizer.latent_token_id)
            
            if stage == 0:
                assert latent_count == 0, f"Stage 0 should have 0 latent tokens, got {latent_count}"
            else:
                expected = min(stage, config.max_latent_stage) * config.c_thought
                assert latent_count == expected, f"Stage {stage} should have {expected} latent tokens, got {latent_count}"
            
            print(f"    ✅ Stage {stage}: {latent_count} latent tokens")
        
        print("  ✅ Stage Manager + Dataset Integration PASSED")
        
    finally:
        shutil.rmtree(temp_dir)


def test_model_trainer_integration():
    """Test Model + Trainer Integration"""
    print("🤖 Testing Model + Trainer Integration...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_simple_test_data(temp_dir)
        
        # Create components
        base_model = SimpleInternVL3Mock()
        tokenizer = SimpleTokenizer()
        
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        config = Config({
            'name': 'integration_test',
            'save_path': temp_dir,
            'batch_size_training': 2,
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'num_workers': 0,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 2,
            'image_size': 224,
            'max_num_patches': 4
        })
        
        trainer = create_multimodal_cot_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Test dataset preparation
        train_loader, val_loader, gen_loader = trainer.prepare_datasets(
            train_data_path=data_path,
            val_data_path=data_path,
            image_root=images_dir
        )
        
        print(f"  ✅ Created data loaders: train={len(train_loader)}, val={len(val_loader)}, gen={len(gen_loader)}")
        
        # Test forward pass
        batch = next(iter(train_loader))
        batch = {k: v for k, v in batch.items() if k != "idx"}
        
        model.train()
        outputs = model(**batch)
        
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)
        print(f"  ✅ Forward pass successful, loss: {outputs.loss.item():.4f}")
        
        # Test optimizer
        optimizer = trainer.setup_optimizer()
        assert optimizer is not None
        print("  ✅ Optimizer setup successful")
        
        print("  ✅ Model + Trainer Integration PASSED")
        
    finally:
        shutil.rmtree(temp_dir)


def test_complete_training_pipeline():
    """Test Complete Training Pipeline"""
    print("🚀 Testing Complete Training Pipeline...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_simple_test_data(temp_dir)
        
        # Setup complete pipeline
        base_model = SimpleInternVL3Mock()
        tokenizer = SimpleTokenizer()
        
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        config = Config({
            'name': 'pipeline_test',
            'save_path': temp_dir,
            'batch_size_training': 1,
            'batch_size_eval': 1,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'num_workers': 0,
            'save_every_n_epochs': 1,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 2,
            'image_size': 224
        })
        
        trainer = create_multimodal_cot_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        
        # Run training
        print("  Running mini training...")
        training_summary = trainer.train(
            train_data_path=data_path,
            val_data_path=data_path,
            image_root=images_dir
        )
        
        # Verify results
        assert 'training_history' in training_summary
        assert len(training_summary['training_history']) == config.num_epochs
        
        # Check checkpoints
        save_dir = Path(training_summary['save_dir'])
        checkpoints = list(save_dir.glob("checkpoint_*"))
        assert len(checkpoints) >= 1
        
        print(f"  ✅ Training completed: {len(training_summary['training_history'])} epochs")
        print(f"  ✅ Checkpoints saved: {len(checkpoints)}")
        print(f"  ✅ Best val loss: {training_summary['best_val_loss']:.4f}")
        
        print("  ✅ Complete Training Pipeline PASSED")
        
    finally:
        shutil.rmtree(temp_dir)


def test_multimodal_data_flow():
    """Test Multimodal Data Flow"""
    print("📊 Testing Multimodal Data Flow...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        data_path, images_dir = create_simple_test_data(temp_dir)
        tokenizer = SimpleTokenizer()
        
        # Test data loading
        dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224
        )
        
        sample = dataset[0]
        
        # Verify multimodal components
        assert 'pixel_values' in sample
        assert 'question_tokenized' in sample
        assert 'steps_tokenized' in sample
        assert 'answer_tokenized' in sample
        assert 'num_patches' in sample
        
        # Check data types and shapes
        pixel_values = sample['pixel_values']
        if isinstance(pixel_values, list):
            pixel_values = torch.tensor(pixel_values)
        
        assert pixel_values.dim() == 4  # [patches, channels, height, width]
        assert pixel_values.shape[1:] == (3, 224, 224)
        
        print(f"  ✅ Image shape: {pixel_values.shape}")
        print(f"  ✅ Question tokens: {len(sample['question_tokenized'])}")
        print(f"  ✅ Steps: {len(sample['steps_tokenized'])}")
        print(f"  ✅ Answer tokens: {len(sample['answer_tokenized'])}")
        
        # Test with model
        base_model = SimpleInternVL3Mock()
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Create a simple batch
        input_ids = torch.tensor([sample['question_tokenized'] + 
                                 sum(sample['steps_tokenized'], []) + 
                                 sample['answer_tokenized']]).long()
        labels = input_ids.clone()
        pixel_values = pixel_values.unsqueeze(0) if pixel_values.dim() == 4 else pixel_values
        
        # Test forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
        
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)
        
        print(f"  ✅ Multimodal forward pass successful")
        print(f"  ✅ Loss: {outputs.loss.item():.4f}")
        
        print("  ✅ Multimodal Data Flow PASSED")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run key integration tests"""
    print("🔑 KEY INTEGRATION TESTS FOR MULTIMODAL COCONUT")
    print("=" * 50)
    
    try:
        test_stage_dataset_integration()
        print()
        
        test_model_trainer_integration()
        print()
        
        test_complete_training_pipeline()
        print()
        
        test_multimodal_data_flow()
        print()
        
        print("🎉 ALL KEY INTEGRATION TESTS PASSED!")
        print("The multimodal CoCoNuT implementation is working correctly.")
        return 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())