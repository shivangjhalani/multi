#!/usr/bin/env python3
"""
Basic Training Example for Multimodal CoCoNuT

This script demonstrates how to set up and run basic training for the Multimodal CoCoNuT model.
It includes configuration loading, model initialization, and training execution.

Usage:
    python examples/basic_training.py --config args/multimodal_coconut.yaml
    python examples/basic_training.py --config args/multimodal_cot.yaml --stage cot
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer
from multimodal_coconut import (
    Config, 
    load_config, 
    MultimodalCoconut,
    setup_logging,
    set_seed,
    init_distributed_training,
    is_main_process
)
from multimodal_coconut.data import ImageProcessor, MultimodalDataset, MultimodalCollator
from multimodal_coconut.training import MultimodalCoTTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Basic Multimodal CoCoNuT Training")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["cot", "coconut"],
        default="coconut",
        help="Training stage: 'cot' for pre-training, 'coconut' for main training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller dataset"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup without actual training"
    )
    
    return parser.parse_args()


def setup_model_and_tokenizer(config: Config):
    """
    Initialize model and tokenizer with proper special tokens
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Add special tokens for CoCoNuT
    special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    try:
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if config.get('fp16', False) else torch.float32
        )
    except Exception as e:
        print(f"Warning: Could not load model {config.model_id}: {e}")
        print("Using mock model for demonstration")
        # Create a simple mock model for demonstration
        base_model = torch.nn.Linear(10, 10)
    
    # Get special token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos_token_id = tokenizer.eos_token_id
    
    # Create MultimodalCoconut model
    model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )
    
    # Resize token embeddings to account for new special tokens
    if hasattr(base_model, 'resize_token_embeddings'):
        base_model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model loaded successfully")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special token IDs: latent={latent_token_id}, start={start_latent_id}, end={end_latent_id}")
    
    return model, tokenizer


def setup_data_pipeline(config: Config, tokenizer, debug: bool = False):
    """
    Set up data loading pipeline
    
    Args:
        config: Configuration object
        tokenizer: Tokenizer instance
        debug: Whether to use debug mode with smaller dataset
        
    Returns:
        Tuple of (train_dataset, val_dataset, image_processor)
    """
    print("Setting up data pipeline...")
    
    # Initialize image processor
    image_processor = ImageProcessor(
        image_size=config.get('image_size', 448),
        max_num_patches=config.get('max_num_patches', 12),
        use_thumbnail=config.get('use_thumbnail', True),
        dynamic_preprocess=config.get('dynamic_preprocess', True)
    )
    
    # Create datasets
    train_dataset = None
    val_dataset = None
    
    if hasattr(config, 'train_data_path') and config.train_data_path != "None":
        if os.path.exists(config.train_data_path):
            train_dataset = MultimodalDataset(
                data_path=config.train_data_path,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_size=100 if debug else config.get('max_train_samples', None),
                image_root=config.get('image_root', None)
            )
            print(f"Training dataset loaded: {len(train_dataset)} samples")
        else:
            print(f"Warning: Training data not found at {config.train_data_path}")
    
    if hasattr(config, 'val_data_path') and config.val_data_path != "None":
        if os.path.exists(config.val_data_path):
            val_dataset = MultimodalDataset(
                data_path=config.val_data_path,
                tokenizer=tokenizer,
                image_processor=image_processor,
                max_size=50 if debug else config.get('max_val_samples', None),
                image_root=config.get('image_root', None)
            )
            print(f"Validation dataset loaded: {len(val_dataset)} samples")
        else:
            print(f"Warning: Validation data not found at {config.val_data_path}")
    
    return train_dataset, val_dataset, image_processor


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Override config based on arguments
    if args.stage == "cot":
        config.cot = True
        config.coconut = False
        print("Running Chain-of-Thought pre-training")
    else:
        config.cot = False
        config.coconut = True
        print("Running CoCoNuT training")
    
    if args.debug:
        config.debug = True
        config.num_epochs = 2
        config.batch_size_training = 2
        config.batch_size_eval = 2
        print("Debug mode enabled")
    
    if args.resume:
        config.resume_from_checkpoint = args.resume
        print(f"Resuming from checkpoint: {args.resume}")
    
    # Set up logging and reproducibility
    setup_logging(config)
    set_seed(config.get('seed', 42))
    
    # Initialize distributed training if needed
    if config.get('use_fsdp', False) or config.get('use_ddp', False):
        init_distributed_training()
    
    # Print configuration summary
    if is_main_process():
        print("\n" + "="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Experiment name: {config.get('name', 'unnamed')}")
        print(f"Model ID: {config.model_id}")
        print(f"Training mode: {'CoT' if config.get('cot', False) else 'CoCoNuT'}")
        print(f"Batch size (train): {config.get('batch_size_training', 8)}")
        print(f"Batch size (eval): {config.get('batch_size_eval', 16)}")
        print(f"Learning rate: {config.get('learning_rate', 1e-5)}")
        print(f"Number of epochs: {config.get('num_epochs', 40)}")
        if config.get('coconut', False):
            print(f"CoCoNuT c_thought: {config.get('c_thought', 2)}")
            print(f"Max latent stage: {config.get('max_latent_stage', 4)}")
            print(f"Epochs per stage: {config.get('epochs_per_stage', 5)}")
        print("="*60)
    
    try:
        # Set up model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Set up data pipeline
        train_dataset, val_dataset, image_processor = setup_data_pipeline(
            config, tokenizer, debug=args.debug
        )
        
        if args.dry_run:
            print("\nDry run completed successfully!")
            print("All components initialized without errors.")
            return
        
        # Create trainer
        trainer = MultimodalCoTTrainer(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            config=config
        )
        
        # Set datasets
        if train_dataset is not None:
            trainer.set_train_dataset(train_dataset)
        if val_dataset is not None:
            trainer.set_val_dataset(val_dataset)
        
        # Start training
        print("\nStarting training...")
        trainer.train()
        
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()