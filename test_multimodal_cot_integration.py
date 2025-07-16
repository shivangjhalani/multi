#!/usr/bin/env python3
"""
Integration test for multimodal CoT training pipeline.

This test demonstrates the complete Stage 0 training pipeline including:
- Model initialization with InternVL3 integration
- Dataset preparation with multimodal data
- Training loop execution
- Validation and checkpointing
- Integration with stage management system
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
from multimodal_coconut.training import MultimodalCoTTrainer, create_multimodal_cot_trainer
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut


class MockInternVL3Model(nn.Module):
    """Mock InternVL3 model for testing"""
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.language_model = MockLanguageModel(vocab_size, hidden_size)
        self.vision_encoder = MockVisionEncoder()
        self.img_context_token_id = 32000  # Mock IMG_CONTEXT token ID
        
        # Mock config
        class MockConfig:
            use_return_dict = True
        self.config = MockConfig()
    
    def extract_feature(self, pixel_values):
        """Mock visual feature extraction"""
        batch_size = pixel_values.shape[0]
        return torch.randn(batch_size, 64)  # Mock visual features
    
    def forward(self, pixel_values=None, input_ids=None, labels=None, **kwargs):
        # For multimodal inputs, use language model with visual features
        if pixel_values is not None:
            # Mock multimodal processing
            visual_features = self.extract_feature(pixel_values)
            # Simple concatenation for testing
            outputs = self.language_model(input_ids=input_ids, labels=labels, **kwargs)
        else:
            # Text-only processing
            outputs = self.language_model(input_ids=input_ids, labels=labels, **kwargs)
        
        return outputs


class MockLanguageModel(nn.Module):
    """Mock language model component"""
    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=4, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Mock config
        class MockConfig:
            hidden_size = hidden_size
            use_return_dict = True
        self.config = MockConfig()
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, 
                attention_mask=None, past_key_values=None, **kwargs):
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        
        # Simple forward pass for testing
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Mock transformer processing
        # Create a simple causal mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Use hidden_states as both source and target for decoder
        memory = hidden_states
        output = self.transformer(hidden_states, memory, tgt_mask=tgt_mask)
        
        logits = self.lm_head(output)
        
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
            def __init__(self, loss, logits, hidden_states=None, past_key_values=None):
                self.loss = loss
                self.logits = logits
                self.hidden_states = [hidden_states] if hidden_states is not None else None
                self.past_key_values = past_key_values
                self.attentions = None
        
        return MockOutput(loss, logits, output, past_key_values)


class MockVisionEncoder(nn.Module):
    """Mock vision encoder"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = self.pool(x)
        return x.flatten(1)


class MockTokenizer:
    """Enhanced mock tokenizer for testing"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.latent_token_id = 101
        self.start_latent_id = 100
        self.end_latent_id = 102
        self.padding_side = "right"
        
        # Mock vocabulary
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<eos>': 2,
            '<img>': 3,
            '</img>': 4,
            '<IMG_CONTEXT>': 32000,
            '<|start-latent|>': 100,
            '<|latent|>': 101,
            '<|end-latent|>': 102,
        }
        
        # Add some common words
        for i, word in enumerate(['the', 'a', 'is', 'in', 'what', 'how', 'why', 'answer'], 5):
            self.vocab[word] = i
    
    def encode(self, text, add_special_tokens=False):
        # Simple mock encoding based on word count
        words = text.split()
        tokens = []
        
        if add_special_tokens:
            tokens.append(1)  # Start token
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Hash-based token ID for consistency
                tokens.append(hash(word) % 900 + 10)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_id):
        for word, id in self.vocab.items():
            if id == token_id:
                return word
        return f"token_{token_id}"


def create_test_data(temp_dir):
    """Create test multimodal dataset"""
    # Create images directory
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create mock images
    image_paths = []
    for i in range(5):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"test_image_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    
    # Create training data
    train_data = []
    for i in range(5):
        sample = {
            "image_path": image_paths[i % len(image_paths)],
            "question": f"What is shown in this image {i}?",
            "steps": [
                f"Step 1: I can see various elements in image {i}",
                f"Step 2: The image shows specific details {i}",
                f"Step 3: Based on the visual information {i}"
            ],
            "answer": f"The answer is description {i}"
        }
        train_data.append(sample)
    
    # Create validation data (smaller)
    val_data = train_data[:3]
    
    # Save JSON files
    train_path = Path(temp_dir) / "train.json"
    val_path = Path(temp_dir) / "val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f)
    
    return str(train_path), str(val_path), str(images_dir)


def test_complete_training_pipeline():
    """Test the complete multimodal CoT training pipeline"""
    print("Testing complete multimodal CoT training pipeline...")
    
    # Create temporary directory for test data
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test data
        train_path, val_path, images_dir = create_test_data(temp_dir)
        
        # Create configuration
        config = Config({
            'name': 'test_multimodal_cot',
            'save_path': temp_dir,
            'model_id': 'mock_internvl3',
            'batch_size_training': 2,
            'batch_size_eval': 2,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_epochs': 2,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'save_every_n_epochs': 1,
            'num_workers': 0,  # Avoid multiprocessing issues in tests
            
            # CoT configuration
            'cot': True,
            'coconut': False,
            
            # CoCoNuT parameters (for compatibility)
            'c_thought': 2,
            'max_latent_stage': 3,
            'epochs_per_stage': 5,
            'uniform_prob': 0.0,
            
            # Multimodal parameters
            'image_size': 224,
            'max_num_patches': 4,
            'use_thumbnail': True,
        })
        
        # Create mock model
        base_model = MockInternVL3Model()
        tokenizer = MockTokenizer()
        
        # Create multimodal CoCoNuT model
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Create trainer
        trainer = create_multimodal_cot_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=0,
            world_size=1
        )
        
        print(f"  Created trainer with config: {config.name}")
        print(f"  Training data: {train_path}")
        print(f"  Validation data: {val_path}")
        print(f"  Images directory: {images_dir}")
        
        # Run training
        print("  Starting training...")
        training_summary = trainer.train(
            train_data_path=train_path,
            val_data_path=val_path,
            image_root=images_dir,
            start_epoch=0
        )
        
        # Verify training completed
        assert 'training_history' in training_summary
        assert len(training_summary['training_history']) == config.num_epochs
        assert 'best_val_loss' in training_summary
        assert 'total_steps' in training_summary
        
        # Check that checkpoints were saved
        save_dir = Path(training_summary['save_dir'])
        assert save_dir.exists()
        
        checkpoint_files = list(save_dir.glob("checkpoint_*"))
        assert len(checkpoint_files) >= 1, f"No checkpoints found in {save_dir}"
        
        best_model_path = save_dir / "best_cot_model"
        assert best_model_path.exists(), f"Best model not saved at {best_model_path}"
        
        # Verify training metrics
        final_metrics = training_summary['training_history'][-1]
        assert 'train_loss' in final_metrics
        assert 'val_loss' in final_metrics
        assert final_metrics['train_loss'] > 0
        assert final_metrics['val_loss'] > 0
        
        print(f"  ✓ Training completed successfully!")
        print(f"  ✓ Final train loss: {final_metrics['train_loss']:.4f}")
        print(f"  ✓ Final val loss: {final_metrics['val_loss']:.4f}")
        print(f"  ✓ Total training steps: {training_summary['total_steps']}")
        print(f"  ✓ Checkpoints saved: {len(checkpoint_files)}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    print("  ✓ Complete training pipeline test passed!")


def test_stage_0_validation():
    """Test that Stage 0 training uses correct configuration"""
    print("Testing Stage 0 configuration validation...")
    
    config = Config({
        'name': 'test_stage_0',
        'save_path': 'test_checkpoints',
        'batch_size_training': 2,
        'learning_rate': 1e-4,
        'num_epochs': 5,
        'epochs_per_stage': 3,
        'max_latent_stage': 2,
        'c_thought': 2,
        
        # These should be corrected for Stage 0
        'cot': False,
        'coconut': True,
    })
    
    base_model = MockInternVL3Model()
    tokenizer = MockTokenizer()
    
    model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=tokenizer.latent_token_id,
        start_latent_id=tokenizer.start_latent_id,
        end_latent_id=tokenizer.end_latent_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    trainer = MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Check that configuration was corrected for Stage 0
    assert trainer.config.cot == True, "CoT should be enabled for Stage 0"
    assert trainer.config.coconut == False, "CoCoNuT should be disabled for Stage 0"
    
    # Check stage manager behavior for CoT mode
    for epoch in [0, 3, 6, 9]:
        stage = trainer.stage_manager.get_current_stage(epoch)
        assert stage == 0, f"Stage 0 training should always return stage 0, got {stage}"
    
    print("  ✓ Stage 0 configuration validation test passed!")


def test_multimodal_data_handling():
    """Test multimodal data handling in the trainer"""
    print("Testing multimodal data handling...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create minimal test data
        train_path, val_path, images_dir = create_test_data(temp_dir)
        
        config = Config({
            'name': 'test_multimodal_data',
            'save_path': temp_dir,
            'batch_size_training': 1,
            'batch_size_eval': 1,
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'num_workers': 0,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 2,
            'image_size': 224,
            'max_num_patches': 4,
        })
        
        base_model = MockInternVL3Model()
        tokenizer = MockTokenizer()
        
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        trainer = MultimodalCoTTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=0,
            world_size=1
        )
        
        # Test dataset preparation
        train_loader, val_loader, gen_loader = trainer.prepare_datasets(
            train_data_path=train_path,
            val_data_path=val_path,
            image_root=images_dir
        )
        
        # Check data loaders
        assert len(train_loader) > 0, "Training loader should not be empty"
        assert len(val_loader) > 0, "Validation loader should not be empty"
        assert len(gen_loader) > 0, "Generation loader should not be empty"
        
        # Test a batch from training loader
        batch = next(iter(train_loader))
        
        # Check batch structure
        required_keys = ['input_ids', 'labels', 'attention_mask', 'pixel_values', 'image_flags']
        for key in required_keys:
            assert key in batch, f"Missing key {key} in batch"
        
        # Check tensor shapes
        assert batch['input_ids'].dim() == 2, "input_ids should be 2D"
        assert batch['labels'].dim() == 2, "labels should be 2D"
        assert batch['pixel_values'].dim() == 4, "pixel_values should be 4D"
        
        # Check that pixel_values has correct shape [total_patches, 3, H, W]
        assert batch['pixel_values'].shape[1] == 3, "pixel_values should have 3 channels"
        
        print(f"  ✓ Batch shapes: input_ids={batch['input_ids'].shape}, "
              f"pixel_values={batch['pixel_values'].shape}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("  ✓ Multimodal data handling test passed!")


def main():
    """Run all integration tests"""
    print("Running multimodal CoT training integration tests...\n")
    
    try:
        test_stage_0_validation()
        print()
        
        test_multimodal_data_handling()
        print()
        
        test_complete_training_pipeline()
        print()
        
        print("🎉 All multimodal CoT training integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())