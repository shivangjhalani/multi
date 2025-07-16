#!/usr/bin/env python3
"""
Debug version of comprehensive test to understand the specific issues
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
from transformers import AutoTokenizer, AutoModel

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
from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut, create_multimodal_coconut_model
from multimodal_coconut.training import (
    StageManager,
    MultimodalCoTTrainer,
    create_stage_manager,
    create_multimodal_cot_trainer
)


def setup_real_model_and_tokenizer():
    """Setup real InternVL model and tokenizer for debugging"""
    try:
        model_id = "OpenGVLab/InternVL3-1B-Pretrained"
        
        print(f"Loading real InternVL model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Add CoCoNuT special tokens
        special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        # Get special token IDs
        start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        
        # Add missing attributes for compatibility
        tokenizer.latent_token_id = latent_id
        tokenizer.start_latent_id = start_latent_id
        tokenizer.end_latent_id = end_latent_id
        
        # Load base model (CPU only for testing)
        base_model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        # Resize token embeddings
        old_vocab_size = base_model.language_model.config.vocab_size
        new_vocab_size = len(tokenizer)
        
        if new_vocab_size > old_vocab_size:
            base_model.language_model.resize_token_embeddings(new_vocab_size)
        
        print(f"✓ Real model loaded successfully")
        print(f"  - Vocab size: {old_vocab_size} -> {new_vocab_size}")
        print(f"  - Special tokens: start={start_latent_id}, latent={latent_id}, end={end_latent_id}")
        
        return base_model, tokenizer
        
    except Exception as e:
        print(f"⚠️  Failed to load real model: {e}")
        return None, None


def create_test_data(temp_dir):
    """Create test dataset"""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create test images
    image_paths = []
    for i in range(4):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = images_dir / f"test_image_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))
    
    # Create training data
    train_data = []
    for i in range(6):
        sample = {
            "image_path": image_paths[i % len(image_paths)],
            "question": f"What is shown in this image number {i}?",
            "steps": [
                f"Step 1: I can see various elements in the image {i}",
                f"Step 2: The image shows specific visual details {i}",
            ],
            "answer": f"The answer is a detailed description of image {i}"
        }
        train_data.append(sample)
    
    # Save datasets
    train_path = Path(temp_dir) / "train.json"
    
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    return str(train_path), str(images_dir)


def debug_model_forward_issue():
    """Debug the model forward pass issue"""
    print("\n🔍 DEBUGGING MODEL FORWARD PASS ISSUE")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Setup model and tokenizer
        base_model, tokenizer = setup_real_model_and_tokenizer()
        
        if base_model is None or tokenizer is None:
            print("❌ Cannot debug without real model")
            return
        
        # Create test data
        train_path, images_dir = create_test_data(temp_dir)
        
        # Create model
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print("✓ Model created successfully")
        
        # Create dataset
        dataset = get_multimodal_dataset(
            data_path=train_path,
            tokenizer=tokenizer,
            image_root=images_dir,
            image_size=224,
            max_num_patches=4,
            max_size=2
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Create collator
        collator = MultimodalCollator(tokenizer=tokenizer, latent_id=tokenizer.latent_token_id)
        
        # Get a sample
        sample = dataset[0]
        print(f"\n📊 Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        # Create batch
        features = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = collator(features)
        
        print(f"\n📦 Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)} - {value}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        # Debug the forward pass issue
        print(f"\n🔍 Debugging forward pass...")
        
        # Check what the base model expects
        print(f"Base model type: {type(base_model)}")
        print(f"Base model forward signature:")
        import inspect
        sig = inspect.signature(base_model.forward)
        print(f"  Parameters: {list(sig.parameters.keys())}")
        
        # Try the forward pass step by step
        print(f"\n🚀 Attempting forward pass...")
        
        try:
            # Remove problematic parameters
            forward_batch = {k: v for k, v in batch.items() if k not in ['idx', 'num_patches_list', '_num_patches_list']}
            
            print(f"Forward batch keys: {list(forward_batch.keys())}")
            
            # Try standard forward first
            outputs = model(**forward_batch)
            print(f"✓ Forward pass successful!")
            print(f"  Loss: {outputs.loss}")
            print(f"  Logits shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Try to understand the issue better
            print(f"\n🔍 Detailed error analysis:")
            import traceback
            traceback.print_exc()
            
            # Try with different parameter combinations
            print(f"\n🧪 Trying different parameter combinations...")
            
            # Try text-only
            try:
                text_only_batch = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask'],
                    'labels': batch.get('labels', None)
                }
                text_outputs = model(**text_only_batch)
                print(f"✓ Text-only forward pass successful!")
            except Exception as text_e:
                print(f"❌ Text-only forward pass failed: {text_e}")
            
            # Try with base model directly
            try:
                base_outputs = base_model(**forward_batch)
                print(f"✓ Base model forward pass successful!")
            except Exception as base_e:
                print(f"❌ Base model forward pass failed: {base_e}")
                print(f"Base model error: {base_e}")
        
    except Exception as e:
        print(f"💥 Critical debugging failure: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def debug_trainer_integration_issue():
    """Debug the trainer integration issue"""
    print("\n🔍 DEBUGGING TRAINER INTEGRATION ISSUE")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Setup model and tokenizer
        base_model, tokenizer = setup_real_model_and_tokenizer()
        
        if base_model is None or tokenizer is None:
            print("❌ Cannot debug without real model")
            return
        
        # Create test data
        train_path, images_dir = create_test_data(temp_dir)
        
        # Create configuration
        config = Config({
            'name': 'debug_test',
            'model_id': 'OpenGVLab/InternVL3-1B-Pretrained',
            'save_path': temp_dir,
            'batch_size_training': 1,  # Use batch size 1 for debugging
            'batch_size_eval': 1,
            'learning_rate': 1e-4,
            'num_epochs': 1,
            'num_workers': 0,
            'save_every_n_epochs': 1,
            'cot': True,
            'coconut': False,
            'c_thought': 2,
            'max_latent_stage': 2,
            'epochs_per_stage': 2,
            'image_size': 224,
            'max_num_patches': 4,
            'pad_latent_to_max': False,
            'uniform_prob': 0.0,
            'no_cot': False,
        })
        
        # Create model
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=tokenizer.latent_token_id,
            start_latent_id=tokenizer.start_latent_id,
            end_latent_id=tokenizer.end_latent_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print("✓ Model created successfully")
        
        # Create trainer
        trainer = create_multimodal_cot_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=0,
            world_size=1
        )
        
        print("✓ Trainer created successfully")
        
        # Debug dataset preparation
        print(f"\n📊 Debugging dataset preparation...")
        
        train_loader, val_loader, gen_loader = trainer.prepare_datasets(
            train_data_path=train_path,
            val_data_path=train_path,  # Use same data for val
            image_root=images_dir
        )
        
        print(f"✓ Datasets prepared successfully")
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader: {len(val_loader)} batches")
        print(f"  Gen loader: {len(gen_loader)} batches")
        
        # Debug single batch
        print(f"\n🔍 Debugging single batch...")
        
        batch = next(iter(train_loader))
        print(f"Batch keys: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: list of length {len(value)} - {value}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
        # Try forward pass with trainer batch
        print(f"\n🚀 Attempting forward pass with trainer batch...")
        
        try:
            # Remove idx and internal parameters if present
            forward_batch = {k: v for k, v in batch.items() if k not in ["idx", "_num_patches_list"]}
            
            model.train()
            outputs = model(**forward_batch)
            print(f"✓ Trainer forward pass successful!")
            print(f"  Loss: {outputs.loss}")
            print(f"  Logits shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"❌ Trainer forward pass failed: {e}")
            print(f"Error type: {type(e)}")
            
            # Detailed error analysis
            print(f"\n🔍 Detailed trainer error analysis:")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"💥 Critical trainer debugging failure: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run debugging tests"""
    print("🔍 MULTIMODAL COCONUT DEBUG SESSION")
    print("=" * 60)
    
    # Debug the model forward pass issue
    debug_model_forward_issue()
    
    # Debug the trainer integration issue
    debug_trainer_integration_issue()
    
    print(f"\n✅ Debugging session completed")


if __name__ == "__main__":
    main()