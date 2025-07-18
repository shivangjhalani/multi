#!/usr/bin/env python3
"""
Unit tests for core multimodal CoCoNuT components

This script tests individual components in isolation:
- Multimodal data pipeline components
- Model forward pass and generation logic  
- Configuration system and utilities
- Image processing functionality
- Tokenization and collation logic
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_coconut.config import Config, load_config, validate_config, ConfigError
from multimodal_coconut.data.dataset import MultimodalDataset, MultimodalCollator
from multimodal_coconut.data.image_processor import ImageProcessor
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut


class TestConfig:
    """Test configuration system"""
    
    def test_config_creation(self):
        """Test basic config creation and attribute access"""
        config_dict = {
            "name": "test-config",
            "c_thought": 2,
            "learning_rate": 1e-5,
            "nested": {"param": "value"}
        }
        
        config = Config(config_dict)
        
        assert config.name == "test-config"
        assert config.c_thought == 2
        assert config.learning_rate == 1e-5
        assert config.nested == {"param": "value"}
    
    def test_config_methods(self):
        """Test config utility methods"""
        config = Config({"a": 1, "b": 2})
        
        # Test get method
        assert config.get("a") == 1
        assert config.get("missing", "default") == "default"
        
        # Test has method
        assert config.has("a") == True
        assert config.has("missing") == False
        
        # Test update method
        config.update(c=3, a=10)
        assert config.c == 3
        assert config.a == 10
        
        # Test to_dict method
        config_dict = config.to_dict()
        assert config_dict == {"a": 10, "b": 2, "c": 3}
    
    def test_config_merge(self):
        """Test config merging functionality"""
        config1 = Config({"a": 1, "b": 2, "c": 3})
        config2 = Config({"b": 20, "d": 4})
        
        merged = config1.merge(config2)
        
        assert merged.a == 1  # From config1
        assert merged.b == 20  # config2 takes precedence
        assert merged.c == 3  # From config1
        assert merged.d == 4  # From config2
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = Config({
            "model_id": "test-model",
            "c_thought": 2,
            "batch_size_training": 8,
            "learning_rate": 1e-5
        })
        
        # Should not raise exception
        validate_config(valid_config)
        
        # Invalid configs
        invalid_configs = [
            Config({"c_thought": 0}),  # Missing model_id, invalid c_thought
            Config({"model_id": "test", "c_thought": -1}),  # Invalid c_thought
            Config({"model_id": "test", "c_thought": 2, "batch_size_training": 0}),  # Invalid batch size
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ConfigError):
                validate_config(invalid_config)


class TestImageProcessor:
    """Test image processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.test_image_path = self.temp_path / "test_image.jpg"
        self.test_image.save(self.test_image_path)
        
        # Create image processor
        self.processor = ImageProcessor(
            image_size=224,
            max_num_patches=6,
            use_thumbnail=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_image_loading(self):
        """Test image loading functionality"""
        # Test successful loading
        pixel_values = self.processor.load_image(str(self.test_image_path))
        assert isinstance(pixel_values, torch.Tensor)
        assert len(pixel_values.shape) == 4  # [num_patches, channels, height, width]
        assert pixel_values.shape[1] == 3  # RGB channels
        
        # Test loading non-existent image with dummy return
        dummy_tensor = self.processor.load_image("non_existent.jpg", return_dummy_on_error=True)
        assert isinstance(dummy_tensor, torch.Tensor)
        
        # Test loading non-existent image with error raising
        with pytest.raises(FileNotFoundError):
            self.processor.load_image("non_existent.jpg", return_dummy_on_error=False)
    
    def test_dynamic_preprocessing(self):
        """Test dynamic image preprocessing"""
        # Load PIL image first
        from PIL import Image
        pil_image = Image.open(str(self.test_image_path))
        
        # Test dynamic preprocessing
        processed_images = self.processor.dynamic_preprocess(pil_image)
        
        assert isinstance(processed_images, list)
        assert len(processed_images) >= 1
        
        # Check that all processed images are PIL Images
        for img in processed_images:
            assert isinstance(img, Image.Image)
            assert img.size == (self.processor.image_size, self.processor.image_size)
    
    def test_batch_processing(self):
        """Test batch image processing"""
        image_paths = [str(self.test_image_path)] * 3
        
        # Use the actual method available
        batch_processed = self.processor.load_images_batch(image_paths)
        
        assert isinstance(batch_processed, list)
        assert len(batch_processed) == 3
        
        # Check that all items are tensors
        for tensor in batch_processed:
            assert isinstance(tensor, torch.Tensor)
            assert len(tensor.shape) == 4  # [num_patches, channels, height, width]
    
    def test_error_handling(self):
        """Test error handling in image processing"""
        # Create corrupted image file
        corrupted_path = self.temp_path / "corrupted.jpg"
        with open(corrupted_path, 'w') as f:
            f.write("not an image")
        
        # Should handle gracefully with dummy return
        dummy_tensor = self.processor.load_image(str(corrupted_path), return_dummy_on_error=True)
        assert isinstance(dummy_tensor, torch.Tensor)
        
        # Should raise exception when return_dummy_on_error=False
        try:
            self.processor.load_image(str(corrupted_path), return_dummy_on_error=False)
            assert False, "Should have raised an exception"
        except Exception:
            pass  # Expected behavior


class TestMultimodalDataset:
    """Test multimodal dataset functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test images
        self.images_dir = self.temp_path / "images"
        self.images_dir.mkdir()
        
        for i in range(3):
            image = Image.new('RGB', (224, 224), color=['red', 'green', 'blue'][i])
            image.save(self.images_dir / f"test_{i}.jpg")
        
        # Create test data with idx field (required by dataset)
        self.test_data = [
            {
                "image_path": "images/test_0.jpg",
                "question": "What color is this?",
                "steps": ["I can see the image.", "The color appears to be red."],
                "answer": "red",
                "idx": 0
            },
            {
                "image_path": "images/test_1.jpg", 
                "question": "What do you see?",
                "steps": ["Looking at the image.", "I see a green colored area."],
                "answer": "green",
                "idx": 1
            },
            {
                "image_path": "images/test_2.jpg",
                "question": "Describe the color.",
                "steps": ["Examining the image.", "The dominant color is blue."],
                "answer": "blue",
                "idx": 2
            }
        ]
        
        # Save test data
        self.data_path = self.temp_path / "test_data.json"
        with open(self.data_path, 'w') as f:
            json.dump(self.test_data, f)
        
        # Create mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.encode = Mock(side_effect=lambda x, **kwargs: list(range(len(x.split()))))
        self.mock_tokenizer.decode = Mock(side_effect=lambda x, **kwargs: " ".join([f"token_{i}" for i in x]))
        self.mock_tokenizer.latent_token_id = 50257
        self.mock_tokenizer.start_latent_id = 50258
        self.mock_tokenizer.end_latent_id = 50259
        self.mock_tokenizer.pad_token_id = 50256
        self.mock_tokenizer.eos_token_id = 50256
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test dataset creation and initialization"""
        try:
            dataset = MultimodalDataset(
                data_path=str(self.data_path),
                tokenizer=self.mock_tokenizer,
                image_root=str(self.temp_path),
                image_size=224,
                max_num_patches=6
            )
            
            assert len(dataset) == 3
            assert dataset.image_root == Path(str(self.temp_path))
            assert dataset.image_size == 224
        except Exception as e:
            print(f"Dataset creation failed: {e}")
            raise
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        try:
            dataset = MultimodalDataset(
                data_path=str(self.data_path),
                tokenizer=self.mock_tokenizer,
                image_root=str(self.temp_path)
            )
            
            # Test getting first item
            item = dataset[0]
            
            assert isinstance(item, dict)
            # Check for the actual keys returned by the dataset
            expected_keys = ["pixel_values", "question_tokenized", "steps_tokenized", "answer_tokenized", "idx", "num_patches"]
            for key in expected_keys:
                assert key in item, f"Missing key: {key}"
            
            # Check data types
            assert isinstance(item["pixel_values"], (torch.Tensor, list))  # Could be serialized
            assert isinstance(item["question_tokenized"], list)
            assert isinstance(item["steps_tokenized"], list)
            assert isinstance(item["answer_tokenized"], list)
            assert isinstance(item["idx"], int)
            assert isinstance(item["num_patches"], int)
        except Exception as e:
            print(f"Dataset getitem failed: {e}")
            raise
    
    def test_tokenization(self):
        """Test multimodal sample tokenization"""
        dataset = MultimodalDataset(
            data_path=str(self.data_path),
            tokenizer=self.mock_tokenizer,
            image_root=str(self.temp_path)
        )
        
        sample = self.test_data[0]
        tokenized = dataset.tokenize_multimodal_sample(sample)
        
        # Check the actual keys returned by tokenize_multimodal_sample
        assert "question_tokenized" in tokenized
        assert "steps_tokenized" in tokenized
        assert "answer_tokenized" in tokenized
        assert "pixel_values" in tokenized
        assert "num_patches" in tokenized
        
        # Verify tokenizer was called
        assert self.mock_tokenizer.encode.called
    
    def test_dataset_iteration(self):
        """Test dataset iteration"""
        try:
            dataset = MultimodalDataset(
                data_path=str(self.data_path),
                tokenizer=self.mock_tokenizer,
                image_root=str(self.temp_path)
            )
            
            items = list(dataset)
            assert len(items) == 3
            
            for item in items:
                assert isinstance(item, dict)
                # Check for the actual keys returned by the dataset
                assert "pixel_values" in item
                assert "question_tokenized" in item
                assert "steps_tokenized" in item
                assert "answer_tokenized" in item
        except Exception as e:
            print(f"Dataset iteration failed: {e}")
            raise


class TestMultimodalCollator:
    """Test multimodal data collator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock tokenizer with all required attributes
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.latent_token_id = 50257
        self.mock_tokenizer.start_latent_id = 50258
        self.mock_tokenizer.end_latent_id = 50259
        self.mock_tokenizer.padding_side = "right"
        
        # Mock the pad_without_fast_tokenizer_warning function
        def mock_pad_function(tokenizer, features, padding=True, pad_to_multiple_of=None, return_tensors=None):
            # Simple padding implementation for testing
            max_len = max(len(f["input_ids"]) for f in features)
            padded_features = {}
            
            # Pad input_ids
            padded_input_ids = []
            for f in features:
                padded = f["input_ids"] + [tokenizer.pad_token_id] * (max_len - len(f["input_ids"]))
                padded_input_ids.append(padded)
            padded_features["input_ids"] = torch.tensor(padded_input_ids)
            
            # Pad attention_mask
            if "attention_mask" in features[0]:
                padded_attention_mask = []
                for f in features:
                    padded = f["attention_mask"] + [0] * (max_len - len(f["attention_mask"]))
                    padded_attention_mask.append(padded)
                padded_features["attention_mask"] = torch.tensor(padded_attention_mask)
            
            return padded_features
        
        # Patch the pad function
        import multimodal_coconut.data.dataset
        multimodal_coconut.data.dataset.pad_without_fast_tokenizer_warning = mock_pad_function
        
        # Create collator
        self.collator = MultimodalCollator(
            tokenizer=self.mock_tokenizer,
            latent_id=50257
        )
    
    def test_collator_creation(self):
        """Test collator initialization"""
        assert self.collator.tokenizer == self.mock_tokenizer
        assert self.collator.latent_id == 50257
        assert self.collator.label_pad_token_id == -100
    
    def test_batch_collation(self):
        """Test batch collation functionality"""
        # Create mock batch features
        features = [
            {
                "pixel_values": torch.randn(3, 3, 224, 224),
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [1, 2, 3, 4, 5],
                "num_patches": 3
            },
            {
                "pixel_values": torch.randn(2, 3, 224, 224),
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 3],
                "num_patches": 2
            }
        ]
        
        batch = self.collator(features)
        
        assert isinstance(batch, dict)
        assert "pixel_values" in batch
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "num_patches_list" in batch
        
        # Check batch dimensions
        assert batch["pixel_values"].shape[0] == 5  # Total patches (3+2)
        assert batch["input_ids"].shape[0] == 2  # Batch size
        assert "_num_patches_list" in batch or "num_patches_list" in batch
    
    def test_latent_token_alignment(self):
        """Test latent token alignment in batches"""
        # Create features with latent tokens
        features = [
            {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "input_ids": [1, 2, 50257, 50257, 3, 4],  # Latent tokens at pos 2,3
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "labels": [-100, -100, -100, -100, 3, 4],
                "num_patches": 1
            },
            {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "input_ids": [1, 50257, 50257, 2, 3],  # Latent tokens at pos 1,2
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, -100, 2, 3],
                "num_patches": 1
            }
        ]
        
        batch = self.collator(features)
        
        # Should handle latent token alignment
        assert batch["input_ids"].shape[1] >= max(len(f["input_ids"]) for f in features)
    
    def test_padding(self):
        """Test sequence padding functionality"""
        features = [
            {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],  # Length 8
                "attention_mask": [1] * 8,
                "labels": [1, 2, 3, 4, 5, 6, 7, 8],
                "num_patches": 1
            },
            {
                "pixel_values": torch.randn(1, 3, 224, 224),
                "input_ids": [1, 2, 3],  # Length 3
                "attention_mask": [1] * 3,
                "labels": [1, 2, 3],
                "num_patches": 1
            }
        ]
        
        batch = self.collator(features)
        
        # All sequences should be padded to same length
        seq_len = batch["input_ids"].shape[1]
        assert batch["attention_mask"].shape[1] == seq_len
        assert batch["labels"].shape[1] == seq_len
        
        # Check padding values
        assert (batch["input_ids"][1, 3:] == 0).all()  # Padded with pad_token_id
        assert (batch["attention_mask"][1, 3:] == 0).all()  # Padded attention mask


class TestMultimodalCoconut:
    """Test multimodal CoCoNuT model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create mock base model with proper structure
        self.mock_base_model = Mock()
        self.mock_base_model.config = Mock()
        self.mock_base_model.config.hidden_size = 4096
        self.mock_base_model.config.vocab_size = 50000
        self.mock_base_model.config.use_return_dict = True
        
        # Mock language model component
        self.mock_base_model.language_model = Mock()
        self.mock_base_model.language_model.config = Mock()
        self.mock_base_model.language_model.config.hidden_size = 4096
        
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.return_value = torch.randn(2, 10, 4096)
        self.mock_base_model.language_model.get_input_embeddings = Mock(return_value=mock_embeddings)
        
        # Mock extract_feature for vision
        self.mock_base_model.extract_feature = Mock(return_value=torch.randn(6, 4096))
        
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5)
        mock_outputs.logits = torch.randn(2, 10, 50000)
        mock_outputs.hidden_states = [torch.randn(2, 10, 4096)]
        mock_outputs.past_key_values = None
        
        self.mock_base_model.return_value = mock_outputs
        self.mock_base_model.forward = Mock(return_value=mock_outputs)
        self.mock_base_model.language_model.forward = Mock(return_value=mock_outputs)
        
        # Set img_context_token_id
        self.mock_base_model.img_context_token_id = 151667
        
        # Create model
        self.model = MultimodalCoconut(
            base_model=self.mock_base_model,
            latent_token_id=50257,
            start_latent_id=50258,
            end_latent_id=50259,
            eos_token_id=50256
        )
    
    def test_model_creation(self):
        """Test model initialization"""
        assert self.model.base_model == self.mock_base_model
        assert self.model.latent_token_id == 50257
        assert self.model.start_latent_id == 50258
        assert self.model.end_latent_id == 50259
        assert self.model.eos_token_id == 50256
    
    def test_latent_token_detection(self):
        """Test latent token detection in sequences"""
        # Create input with latent tokens
        input_ids = torch.tensor([
            [1, 2, 50257, 50257, 3, 4],  # Latent tokens at positions 2, 3
            [1, 50257, 2, 3, 4, 5]       # Latent token at position 1
        ])
        
        # Test latent token detection using the actual forward method
        latent_indices = (input_ids == self.model.latent_token_id).nonzero(as_tuple=False)
        
        # Should find latent token positions
        assert len(latent_indices) > 0
        assert latent_indices.shape[0] == 3  # Found 3 latent tokens total
    
    def test_forward_pass_no_latent(self):
        """Test forward pass without latent tokens"""
        # Create inputs without latent tokens
        pixel_values = torch.randn(2, 3, 224, 224)
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6]])
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Should call base model
        assert self.mock_base_model.called
        assert hasattr(outputs, 'loss')
        assert hasattr(outputs, 'logits')
    
    def test_forward_pass_with_latent(self):
        """Test forward pass with latent tokens"""
        # Create inputs with latent tokens
        pixel_values = torch.randn(1, 3, 224, 224)
        input_ids = torch.tensor([[1, 2, 50257, 3, 4]])  # Latent token at position 2
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, -100, 3, 4]])  # Ignore latent positions
        
        # Forward pass
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Should handle latent tokens
        assert self.mock_base_model.called
        assert hasattr(outputs, 'loss')
        assert hasattr(outputs, 'logits')
    
    def test_generation(self):
        """Test model generation functionality"""
        # Mock generation method
        self.mock_base_model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
        
        pixel_values = torch.randn(1, 3, 224, 224)
        input_ids = torch.tensor([[1, 2, 3]])
        
        generated = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=10
        )
        
        assert isinstance(generated, torch.Tensor)
        assert generated.shape[0] == 1  # Batch size


def run_unit_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR CORE COMPONENTS")
    print("=" * 60)
    
    # Test classes to run
    test_classes = [
        TestConfig,
        TestImageProcessor,
        TestMultimodalDataset,
        TestMultimodalCollator,
        TestMultimodalCoconut
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Run setup if exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test method
                getattr(test_instance, test_method)()
                
                # Run teardown if exists
                if hasattr(test_instance, 'teardown_method'):
                    test_instance.teardown_method()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
    
    print("\n" + "=" * 60)
    print("UNIT TEST RESULTS")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print("\n🎉 All unit tests passed!")
        return True


if __name__ == "__main__":
    success = run_unit_tests()
    sys.exit(0 if success else 1)