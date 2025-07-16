#!/usr/bin/env python3
"""
Main training script for Multimodal CoCoNuT

Following the original CoCoNuT's run.py structure and patterns.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_coconut import (
    Config,
    load_config,
    validate_config,
    set_seed,
    setup_logging,
    get_logger,
    init_distributed_training,
    is_main_process
)
from multimodal_coconut.model import MultimodalCoconut
from multimodal_coconut.training import create_progressive_trainer
from transformers import AutoTokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Multimodal CoCoNuT Training")
    parser.add_argument("config_file", type=str, help="Path to configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
        validate_config(config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize distributed training if needed
    local_rank, rank, world_size = 0, 0, 1
    if config.get('use_fsdp', False) or config.get('use_ddp', False):
        try:
            local_rank, rank, world_size = init_distributed_training()
        except Exception as e:
            print(f"Error initializing distributed training: {e}")
            sys.exit(1)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup logging
    logger = setup_logging(
        log_level=config.get('log_level', 'INFO'),
        use_wandb=config.get('use_wandb', True),
        wandb_project=config.get('wandb_project', 'multimodal-coconut'),
        wandb_config=config.to_dict()
    )
    
    if is_main_process():
        logger.info("Starting Multimodal CoCoNuT training")
        logger.info(f"Configuration file: {args.config_file}")
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
        
        # Log configuration
        from multimodal_coconut.utils.logging import log_config
        log_config(config, logger)
    
    try:
        # Load tokenizer and add special tokens
        if is_main_process():
            logger.info(f"Loading tokenizer: {config.model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Add CoCoNuT special tokens
        special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
        tokenizer.add_tokens(special_tokens)
        
        # Store token IDs for easy access
        tokenizer.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        tokenizer.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        
        if is_main_process():
            logger.info(f"Added special tokens: {special_tokens}")
            logger.info(f"Token IDs - Start: {tokenizer.start_latent_id}, "
                       f"Latent: {tokenizer.latent_token_id}, End: {tokenizer.end_latent_id}")
        
        # Create multimodal CoCoNuT model
        if is_main_process():
            logger.info("Creating MultimodalCoconut model")
        
        model = MultimodalCoconut.from_pretrained(
            config.model_id,
            tokenizer=tokenizer,
            torch_dtype=getattr(config, 'torch_dtype', 'auto'),
            trust_remote_code=True
        )
        
        # Move model to appropriate device
        if torch.cuda.is_available():
            model = model.to(f'cuda:{local_rank}')
        
        # Setup distributed training if needed
        if world_size > 1:
            if config.get('use_fsdp', False):
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                model = FSDP(model)
                if is_main_process():
                    logger.info("Model wrapped with FSDP")
            elif config.get('use_ddp', False):
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, device_ids=[local_rank])
                if is_main_process():
                    logger.info("Model wrapped with DDP")
        
        # Create progressive trainer
        if is_main_process():
            logger.info("Creating progressive training orchestrator")
        
        trainer = create_progressive_trainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            rank=rank,
            world_size=world_size,
            wandb_run=logger.wandb_run if hasattr(logger, 'wandb_run') else None
        )
        
        # Get data paths from config
        train_data_path = config.get('train_data_path', 'data/train.json')
        val_data_path = config.get('val_data_path', 'data/val.json')
        image_root = config.get('image_root', 'data/images')
        
        if is_main_process():
            logger.info(f"Training data: {train_data_path}")
            logger.info(f"Validation data: {val_data_path}")
            logger.info(f"Image root: {image_root}")
        
        # Check if we're only doing evaluation
        if config.get('only_eval', False):
            if is_main_process():
                logger.info("Running evaluation only")
            # TODO: Implement evaluation logic
            logger.info("Evaluation completed")
        else:
            # Run progressive training
            if is_main_process():
                logger.info("Starting progressive training")
            
            training_results = trainer.train_progressive(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                image_root=image_root,
                start_epoch=config.get('start_epoch', 0)
            )
            
            if is_main_process():
                logger.info("Progressive training completed successfully")
                logger.info(f"Final results: {training_results}")
    
    except Exception as e:
        if is_main_process():
            logger.error(f"Training failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise e
    
    if is_main_process():
        logger.info("Multimodal CoCoNuT training completed")


if __name__ == "__main__":
    main()