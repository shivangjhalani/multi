#!/usr/bin/env python3
"""
Comprehensive Integration Test for Multimodal CoCoNuT

This test validates the entire multimodal CoCoNuT implementation including:
- Configuration system
- Multimodal data pipeline (dataset, collator, image processing)
- Stage management system
- Multimodal CoCoNuT model (forward pass, generation, continuous thoughts)
- CoT trainer (Stage 0 training)
- Integration between all components
- End-to-end training and inference pipeline

This is the definitive test to ensure all components work together correctly.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset

# Import all multimodal CoCoNuT components
from multimodal_coconut.config import Config, load_config, validate_config
from multimodal_coconut.data.dataset import (
    MultimodalDataset, 
    MultimodalCollator,
    get_multimodal_dataset,
    get_multimodal_cot_latent_dataset,
    get_multimodal_question_latent_dataset
)
from multimodal_coconut.data.image_processor import ImageProcessor
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from multimodal_coconut.training import (
    StageManager,
    MultimodalCoTTrainer,
    create_stage_manager,
    create_multimodal_cot_trainer
)


class TestResults:
    """Track test results across all components"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record_pass(self, test_name):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, str(error)))
        print(f"  ❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        
        if self.failed > 0:
            print(f"\nFailed tests:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        return self.failed == 0


# Mock components for testing
class MockInternVL3Model(nn.Module):
    """Comprehensive mock InternVL3 model"""    def __in
it__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.language_model = MockLanguageModel(vocab_size, hidden_size)
        self.vision_encoder = MockVisionEncoder()
        self.img_context_token_id = 32000
        
        class MockConfig:
            use_return_dict = True
        self.config = MockConfig()
    
    def extract_feature(self, pixel_values):
        """Extract visual features from images"""
        if pixel_values is None:
            return None
        batch_size = pixel_values.shape[0]
        return torch.randn(batch_size, 128)
    
    def forward(self, pixel_values=None, input_ids=None, labels=None, **kwargs):
        if pixel_values is not None:
            visual_features = self.extract_feature(pixel_values)
        return self.language_model(input_ids=input_ids, labels=labels, **kwargs)


class MockLanguageModel(nn.Module):
    """Mock language model with proper transformer architecture"""
    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_size, nhead=8, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        class MockConfig:
            hidden_size = hidden_size
            use_return_dict = True
        self.config = MockConfig()
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def forward(self, input_ids=None, inputs_embeds=None, labels=None, 
                attention_mask=None, past_key_values=None, output_hidden_states=False, **kwargs):
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create causal mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Simple transformer processing
        memory = hidden_states
        output = self.transformer(hidden_states, memory, tgt_mask=tgt_mask)
        logits = self.lm_head(output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        class MockOutput:
            def __init__(self, loss, logits, hidden_states=None, past_key_values=None):
                self.loss = loss
                self.logits = logits
                self.hidden_states = [hidden_states] if output_hidden_states else None
                self.past_key_values = past_key_values
                self.attentions = None
        
        return MockOutput(loss, logits, output, past_key_values)


class MockVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, pixel_values):
        x = self.conv(pixel_values)
        x = self.pool(x)
        return x.flatten(1)


class MockTokenizer:
    """Enhanced mock tokenizer with full vocabulary"""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.latent_token_id = 101
        self.start_latent_id = 100
        self.end_latent_id = 102
        self.padding_side = "right"
        
        # Comprehensive vocabulary
        self.vocab = {
            '<pad>': 0, '<unk>': 1, '<eos>': 2, '<img>': 3, '</img>': 4,
            '<IMG_CONTEXT>': 32000, '<|start-latent|>': 100, '<|latent|>': 101, '<|end-latent|>': 102,
            'what': 10, 'is': 11, 'in': 12, 'the': 13, 'image': 14, 'answer': 15, 'step': 16,
            'this': 17, 'shows': 18, 'can': 19, 'see': 20, 'based': 21, 'on': 22, 'visual': 23
        }
    
    def encode(self, text, add_special_tokens=False):
        words = text.lower().split()
        tokens = []
        
        if add_special_tokens:
            tokens.append(1)
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(hash(word) % 900 + 50)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_id):
        for word, id in self.vocab.items():
            if id == token_id:
                return word
        return f"token_{token_id}"


def create_comprehensive_test_data(temp_dir):
    """Create comprehensive test dataset with realistic multimodal data"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create diverse test images
    image_paths = []
    for i in range(8):
        # Create varied test images
        if i % 2 == 0:
            img_array = np.random.randint(100, 200, (448, 448, 3), dtype=np.uint8)
        else:
            img_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
        
        img = Image.fromarray(img_array)
        img_path = images_dir / f"test_image_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    
    # Create comprehensive training data
    train_data = []
    for i in range(12):
        sample = {
            "image_path": image_paths[i % len(image_paths)],
            "question": f"What is shown in this image number {i}?",
            "steps": [
                f"Step 1: I can see various elements in the image {i}",
                f"Step 2: The image shows specific visual details {i}",
                f"Step 3: Based on the visual information, I can determine {i}",
                f"Step 4: The key features indicate {i}"
            ],
            "answer": f"The answer is a detailed description of image {i}"
        }
        train_data.append(sample)
    
    # Create validation data
    val_data = train_data[:6]
    
    # Save datasets
    train_path = Path(temp_dir) / "train.json"
    val_path = Path(temp_dir) / "val.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    return str(train_path), str(val_path), str(images_dir)


def test_configuration_system(results):
    """Test the configuration system comprehensively"""
    print("\n🔧 Testing Configuration System...")
    
    try:
        # Test basic config creation
        config_dict = {
            'name': 'test_config',
            'model_id': 'test_model',
            'c_thought': 2,
            'max_latent_stage': 4,
            'batch_size_training': 8,
            'learning_rate': 1e-5
        }
        
        config = Config(config_dict)
        assert config.name == 'test_config'
        assert config.c_thought == 2
        results.record_pass("Basic config creation")
        
        # Test config validation
        validate_config(config)
        results.record_pass("Config validation")
        
        # Test config updates
        config.update(learning_rate=2e-5, new_param='test')
        assert config.learning_rate == 2e-5
        assert config.new_param == 'test'
        results.record_pass("Config updates")
        
        # Test to_dict conversion
        config_dict_back = config.to_dict()
        assert isinstance(config_dict_back, dict)
        assert config_dict_back['learning_rate'] == 2e-5
        results.record_pass("Config to_dict conversion")
        
    except Exception as e:
        results.record_fail("Configuration system", e)


def test_image_processing_pipeline(results, temp_dir):
    """Test the image processing pipeline"""
    print("\n🖼️  Testing Image Processing Pipeline...")
    
    try:
        # Create test image
        img_array = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = Path(temp_dir) / "test_image.jpg"
        img.save(img_path)
        
        # Test ImageProcessor
        processor = ImageProcessor(image_size=448, max_num_patches=12, use_thumbnail=True)
        
        # Test image loading
        pixel_values = processor.load_image(str(img_path))
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.dim() == 4  # [num_patches, 3, H, W]
        assert pixel_values.shape[1:] == (3, 448, 448)
        results.record_pass("Image loading and preprocessing")
        
        # Test error handling
        pixel_values_dummy = processor.load_image("nonexistent.jpg", return_dummy_on_error=True)
        assert isinstance(pixel_values_dummy, torch.Tensor)
        results.record_pass("Image error handling")
        
    except Exception as e:
        results.record_fail("Image processing pipeline", e)


def test_multimodal_dataset_pipeline(results, temp_dir):
    """Test the complete multimodal dataset pipeline"""
    print("\n📊 Testing Multimodal Dataset Pipeline...")
    
    try:
        train_path, val_path, images_dir = create_comprehensive_test_data(temp_dir)
        tokenizer = MockTokenizer()
        
        # Test MultimodalDataset
        dataset = MultimodalDataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_num_patches=4
        )
        
        assert len(dataset) > 0
        sample = dataset[0]
        
        # Verify sample structure
        required_keys = ['pixel_values', 'question_tokenized', 'steps_tokenized', 'answer_tokenized', 'num_patches']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
        
        results.record_pass("MultimodalDataset creation and sampling")
        
        # Test dataset verification
        verification_result = dataset.verify_sample(0)
        assert verification_result == True
        results.record_pass("Dataset sample verification")
        
        # Test get_multimodal_dataset convenience function
        hf_dataset = get_multimodal_dataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_size=5
        )
        
        assert len(hf_dataset) <= 5
        results.record_pass("get_multimodal_dataset function")
        
    except Exception as e:
        results.record_fail("Multimodal dataset pipeline", e)


def test_multimodal_collator(results, temp_dir):
    """Test the multimodal data collator"""
    print("\n🔄 Testing Multimodal Collator...")
    
    try:
        train_path, val_path, images_dir = create_comprehensive_test_data(temp_dir)
        tokenizer = MockTokenizer()
        
        # Create dataset
        dataset = get_multimodal_dataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_size=4
        )
        
        # Test collator
        collator = MultimodalCollator(tokenizer=tokenizer, latent_id=tokenizer.latent_token_id)
        
        # Create batch
        features = [dataset[i] for i in range(min(3, len(dataset)))]
        batch = collator(features)
        
        # Verify batch structure
        required_keys = ['input_ids', 'attention_mask', 'pixel_values', 'image_flags', 'num_patches_list']
        for key in required_keys:
            assert key in batch, f"Missing batch key: {key}"
        
        # Verify tensor shapes
        assert batch['input_ids'].dim() == 2
        assert batch['pixel_values'].dim() == 4
        assert batch['image_flags'].shape[0] == len(features)
        
        results.record_pass("MultimodalCollator basic functionality")
        
        # Test with latent tokens (simulate CoCoNuT data)
        config = Config({'c_thought': 2, 'max_latent_stage': 2, 'uniform_prob': 0.0, 'no_cot': False, 'pad_latent_to_max': False})
        
        cot_dataset = get_multimodal_cot_latent_dataset(
            scheduled_stage=1,
            base_dataset=dataset,
            configs=config,
            start_id=tokenizer.start_latent_id,
            latent_id=tokenizer.latent_token_id,
            end_id=tokenizer.end_latent_id,
            no_special_marker=False
        )
        
        cot_features = [cot_dataset[i] for i in range(min(2, len(cot_dataset)))]
        cot_batch = collator(cot_features)
        
        # Verify latent token alignment
        assert 'position_ids' in cot_batch
        results.record_pass("MultimodalCollator with latent tokens")
        
    except Exception as e:
        results.record_fail("Multimodal collator", e)


def test_stage_management_system(results):
    """Test the stage management system comprehensively"""
    print("\n📈 Testing Stage Management System...")
    
    try:
        config = Config({
            'epochs_per_stage': 3,
            'max_latent_stage': 4,
            'c_thought': 2,
            'uniform_prob': 0.1,
            'cot': False,
            'coconut': True,
            'no_cot': False,
            'pad_latent_to_max': False
        })
        
        stage_manager = StageManager(config)
        
        # Test stage calculation
        assert stage_manager.get_current_stage(0) == 0
        assert stage_manager.get_current_stage(3) == 1
        assert stage_manager.get_current_stage(6) == 2
        results.record_pass("Stage calculation")
        
        # Test stage info
        stage_0 = stage_manager.get_stage_info(0)
        assert stage_0.is_cot_stage == True
        assert stage_0.num_latent_tokens == 0
        
        stage_2 = stage_manager.get_stage_info(2)
        assert stage_2.is_cot_stage == False
        assert stage_2.num_latent_tokens == 4  # 2 steps * 2 c_thought
        results.record_pass("Stage info generation")
        
        # Test effective stage calculation
        sample_steps = ["Step 1", "Step 2", "Step 3"]
        effective_stage, n_skip, n_latent = stage_manager.get_effective_stage_for_sample(2, sample_steps)
        assert effective_stage == 2
        assert n_skip == 2
        assert n_latent == 4
        results.record_pass("Effective stage calculation")
        
        # Test curriculum summary
        summary = stage_manager.get_training_summary(10)
        assert 'stages' in summary
        assert len(summary['stages']) > 0
        results.record_pass("Curriculum summary generation")
        
    except Exception as e:
        results.record_fail("Stage management system", e)


def test_multimodal_coconut_model(results):
    """Test the multimodal CoCoNuT model"""
    print("\n🥥 Testing Multimodal CoCoNuT Model...")
    
    try:
        # Create model components
        base_model = MockInternVL3Model()
        tokenizer = MockTokenizer()
        
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Test model initialization
        assert model.latent_token_id == tokenizer.latent_token_id
        assert model.hidden_size > 0
        results.record_pass("Model initialization")
        
        # Test standard forward pass (no latent tokens)
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        labels = torch.randint(1, 100, (batch_size, seq_len))
        pixel_values = torch.randn(4, 3, 224, 224)  # 2 samples * 2 patches each
        attention_mask = torch.ones(batch_size, seq_len)
        image_flags = torch.ones(batch_size, 1)
        
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_flags=image_flags
        )
        
        assert outputs.loss is not None
        assert outputs.logits is not None
        assert not torch.isnan(outputs.loss)
        results.record_pass("Standard multimodal forward pass")
        
        # Test forward pass with latent tokens
        latent_input_ids = input_ids.clone()
        latent_input_ids[:, 3:5] = tokenizer.latent_token_id  # Add latent tokens
        
        latent_outputs = model(
            pixel_values=pixel_values,
            input_ids=latent_input_ids,
            labels=labels,
            attention_mask=attention_mask,
            image_flags=image_flags
        )
        
        assert latent_outputs.loss is not None
        assert latent_outputs.logits is not None
        results.record_pass("Forward pass with latent tokens")
        
        # Test text-only processing
        text_outputs = model(
            pixel_values=None,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        
        assert text_outputs.loss is not None
        results.record_pass("Text-only processing")
        
        # Test generation
        gen_outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids[:, :5],  # Shorter input for generation
            attention_mask=attention_mask[:, :5],
            image_flags=image_flags,
            generation_config={'max_new_tokens': 5, 'do_sample': False}
        )
        
        assert gen_outputs is not None
        assert gen_outputs.shape[0] == batch_size
        results.record_pass("Multimodal generation")
        
    except Exception as e:
        results.record_fail("Multimodal CoCoNuT model", e)


def test_cot_trainer_integration(results, temp_dir):
    """Test the CoT trainer integration"""
    print("\n🎓 Testing CoT Trainer Integration...")
    
    try:
        # Create test data
        train_path, val_path, images_dir = create_comprehensive_test_data(temp_dir)
        
        # Create configuration
        config = Config({
            'name': 'test_integration',
            'save_path': temp_dir,
            'batch_size_training': 2,
            'batch_size_eval': 2,
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'num_workers': 0,
            'save_every_n_epochs': 1,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 2,
            'image_size': 224,
            'max_num_patches': 4,
        })
        
        # Create model
        base_model = MockInternVL3Model()
        tokenizer = MockTokenizer()
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
        
        results.record_pass("CoT trainer creation")
        
        # Test dataset preparation
        train_loader, val_loader, gen_loader = trainer.prepare_datasets(
            train_data_path=train_path,
            val_data_path=val_path,
            image_root=images_dir
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(gen_loader) > 0
        results.record_pass("Dataset preparation")
        
        # Test optimizer setup
        optimizer = trainer.setup_optimizer()
        assert optimizer is not None
        results.record_pass("Optimizer setup")
        
        # Test single training step
        model.train()
        batch = next(iter(train_loader))
        
        # Move batch to device (CPU for testing)
        batch = {k: v for k, v in batch.items() if k != "idx"}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        assert loss is not None
        assert not torch.isnan(loss)
        results.record_pass("Single training step")
        
        # Test validation step
        model.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_batch = {k: v for k, v in val_batch.items() if k != "idx"}
            val_outputs = model(**val_batch)
            val_loss = val_outputs.loss
            
            assert val_loss is not None
            assert not torch.isnan(val_loss)
        results.record_pass("Single validation step")
        
    except Exception as e:
        results.record_fail("CoT trainer integration", e)


def test_end_to_end_pipeline(results, temp_dir):
    """Test the complete end-to-end pipeline"""
    print("\n🚀 Testing End-to-End Pipeline...")
    
    try:
        # Create comprehensive test setup
        train_path, val_path, images_dir = create_comprehensive_test_data(temp_dir)
        
        config = Config({
            'name': 'end_to_end_test',
            'save_path': temp_dir,
            'batch_size_training': 2,
            'batch_size_eval': 2,
            'learning_rate': 1e-4,
            'num_epochs': 2,
            'num_workers': 0,
            'save_every_n_epochs': 1,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 3,
            'epochs_per_stage': 2,
            'image_size': 224,
            'max_num_patches': 4,
        })
        
        # Full pipeline test
        base_model = MockInternVL3Model()
        tokenizer = MockTokenizer()
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        trainer = create_multimodal_cot_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=0,
            world_size=1
        )
        
        # Run mini training
        training_summary = trainer.train(
            train_data_path=train_path,
            val_data_path=val_path,
            image_root=images_dir
        )
        
        # Verify training completed
        assert 'training_history' in training_summary
        assert len(training_summary['training_history']) == config.num_epochs
        assert 'best_val_loss' in training_summary
        
        # Check checkpoints
        save_dir = Path(training_summary['save_dir'])
        checkpoint_files = list(save_dir.glob("checkpoint_*"))
        assert len(checkpoint_files) >= 1
        
        results.record_pass("Complete end-to-end training pipeline")
        
        # Test different stages
        stage_manager = StageManager(config)
        
        # Test Stage 0 (CoT)
        stage_0_info = stage_manager.get_stage_info(0)
        assert stage_0_info.is_cot_stage == True
        
        # Test Stage 1 (First CoCoNuT stage)
        stage_1_info = stage_manager.get_stage_info(1)
        assert stage_1_info.is_cot_stage == False
        assert stage_1_info.num_latent_tokens == 2
        
        results.record_pass("Multi-stage curriculum validation")
        
    except Exception as e:
        results.record_fail("End-to-end pipeline", e)


def main():
    """Run comprehensive integration tests"""
    print("🧪 COMPREHENSIVE MULTIMODAL COCONUT INTEGRATION TEST")
    print("=" * 60)
    print("Testing all components and their integration...")
    
    results = TestResults()
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Run all test suites
        test_configuration_system(results)
        test_image_processing_pipeline(results, temp_dir)
        test_multimodal_dataset_pipeline(results, temp_dir)
        test_multimodal_collator(results, temp_dir)
        test_stage_management_system(results)
        test_multimodal_coconut_model(results)
        test_cot_trainer_integration(results, temp_dir)
        test_end_to_end_pipeline(results, temp_dir)
        
        # Print final results
        success = results.summary()
        
        if success:
            print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("The multimodal CoCoNuT implementation is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please review the errors above.")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n💥 Critical test failure: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    exit(main())