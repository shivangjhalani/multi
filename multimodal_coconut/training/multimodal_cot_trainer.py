"""
Multimodal CoT Pre-training (Stage 0) Implementation

This module implements the training loop for standard multimodal chain-of-thought,
which serves as the foundation for the CoCoNuT curriculum. Stage 0 trains the model
to perform explicit multimodal reasoning steps before transitioning to continuous thoughts.

Key features:
- Standard multimodal chain-of-thought training (no latent tokens)
- Loss calculation for multimodal reasoning steps
- Validation logic for CoT pre-training stage
- Integration with InternVL3 and multimodal data pipeline
- Follows original CoCoNuT training patterns exactly
"""

import os
import gc
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..config import Config
from ..data.dataset import (
    get_multimodal_dataset,
    get_multimodal_cot_latent_dataset,
    get_multimodal_question_latent_dataset,
    MultimodalCollator
)
from .stage_manager import StageManager


class MultimodalCoTTrainer:
    """
    Trainer for multimodal Chain-of-Thought pre-training (Stage 0).
    
    This trainer implements the foundation stage of the CoCoNuT curriculum,
    where the model learns to perform explicit multimodal reasoning steps
    before transitioning to continuous thoughts in later stages.
    
    Following the original CoCoNuT pattern exactly:
    - Stage 0: cot=True, coconut=False
    - Standard autoregressive language modeling loss
    - Full supervision on reasoning steps and answers
    - No latent tokens in Stage 0
    """
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 config: Config,
                 rank: int = 0,
                 world_size: int = 1,
                 wandb_run=None):
        """
        Initialize multimodal CoT trainer
        
        Args:
            model: Multimodal CoCoNuT model (will be used in standard mode for Stage 0)
            tokenizer: Tokenizer with special tokens added
            config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
            wandb_run: Weights & Biases run for logging (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.wandb_run = wandb_run
        
        # Stage manager for curriculum
        self.stage_manager = StageManager(config)
        
        # Special token IDs
        self.latent_id = getattr(tokenizer, 'latent_token_id', None)
        self.start_id = getattr(tokenizer, 'start_latent_id', None)
        self.end_id = getattr(tokenizer, 'end_latent_id', None)
        
        # Training state
        self.current_epoch = 0
        self.total_train_steps = 0
        self.best_val_loss = float('inf')
        
        # Setup directories
        self.save_dir = Path(config.save_path) / config.name
        if rank == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Collator for multimodal data
        self.collator = MultimodalCollator(
            tokenizer=tokenizer,
            latent_id=self.latent_id,
            label_pad_token_id=-100
        )
        
        # Validate configuration for CoT training
        self._validate_cot_config()
    
    def _validate_cot_config(self):
        """Validate configuration for CoT pre-training"""
        if not hasattr(self.config, 'cot') or not self.config.cot:
            print("Warning: CoT flag not set to True. Setting it for Stage 0 training.")
            self.config.cot = True
        
        if hasattr(self.config, 'coconut') and self.config.coconut:
            print("Warning: CoCoNuT flag set to True. Disabling it for Stage 0 training.")
            self.config.coconut = False
        
        # Ensure required parameters exist
        required_params = ['batch_size_training', 'learning_rate', 'num_epochs']
        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing required config parameter: {param}")
    
    def prepare_datasets(self, 
                        train_data_path: str,
                        val_data_path: str,
                        image_root: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare training and validation datasets for CoT pre-training.
        
        Args:
            train_data_path: Path to training data JSON file
            val_data_path: Path to validation data JSON file  
            image_root: Root directory for images
            
        Returns:
            Tuple of (train_dataloader, val_loss_dataloader, val_gen_dataloader)
        """
        print(f"Rank {self.rank}: Preparing multimodal CoT datasets...")
        
        # Load base datasets
        base_dataset_train = get_multimodal_dataset(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            image_root=image_root,
            image_size=getattr(self.config, 'image_size', 448),
            max_num_patches=getattr(self.config, 'max_num_patches', 12),
            use_thumbnail=getattr(self.config, 'use_thumbnail', True),
            max_size=getattr(self.config, 'max_train_samples', 1000000000)
        )
        
        base_dataset_valid = get_multimodal_dataset(
            data_path=val_data_path,
            tokenizer=self.tokenizer,
            image_root=image_root,
            image_size=getattr(self.config, 'image_size', 448),
            max_num_patches=getattr(self.config, 'max_num_patches', 12),
            use_thumbnail=getattr(self.config, 'use_thumbnail', True),
            max_size=getattr(self.config, 'max_val_samples', 1000000000)
        )
        
        # For Stage 0 (CoT), scheduled_stage = 0 (no latent tokens)
        scheduled_stage = 0
        
        # Prepare training dataset (Stage 0 = standard CoT)
        dataset_train = get_multimodal_cot_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset=base_dataset_train,
            configs=self.config,
            start_id=self.start_id,
            latent_id=self.latent_id,
            end_id=self.end_id,
            no_special_marker=True,  # No special markers for CoT
            shuffle=True
        )
        
        # Prepare validation dataset for loss calculation
        dataset_loss_val = get_multimodal_cot_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset=base_dataset_valid,
            configs=self.config,
            start_id=self.start_id,
            latent_id=self.latent_id,
            end_id=self.end_id,
            no_special_marker=True,  # No special markers for CoT
            shuffle=False
        )
        
        # Prepare validation dataset for generation
        dataset_gen_val = get_multimodal_question_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset_valid=base_dataset_valid,
            configs=self.config,
            start_id=self.start_id,
            latent_id=self.latent_id,
            end_id=self.end_id,
            no_special_marker=True  # No special markers for CoT
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            dataset_train,
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=False,  # Shuffling handled by DistributedSampler
            pin_memory=True,
            batch_size=self.config.batch_size_training,
            collate_fn=self.collator,
            sampler=DistributedSampler(dataset_train, shuffle=True) if self.world_size > 1 else None
        )
        
        val_loss_dataloader = DataLoader(
            dataset_loss_val,
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=False,
            pin_memory=True,
            batch_size=getattr(self.config, 'batch_size_eval', self.config.batch_size_training),
            collate_fn=self.collator,
            sampler=DistributedSampler(dataset_loss_val, shuffle=False) if self.world_size > 1 else None
        )
        
        val_gen_dataloader = DataLoader(
            dataset_gen_val,
            num_workers=getattr(self.config, 'num_workers', 1),
            pin_memory=True,
            batch_size=1,  # Generation typically done with batch_size=1
            collate_fn=self.collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False) if self.world_size > 1 else None
        )
        
        if self.rank == 0:
            print(f"Training samples: {len(dataset_train)}")
            print(f"Validation samples: {len(dataset_loss_val)}")
            print(f"Generation samples: {len(dataset_gen_val)}")
        
        return train_dataloader, val_loss_dataloader, val_gen_dataloader
    
    def setup_optimizer(self) -> optim.Optimizer:
        """
        Setup optimizer for CoT pre-training.
        
        Returns:
            Configured optimizer
        """
        # Get optimizer parameters
        learning_rate = getattr(self.config, 'learning_rate', 1e-5)
        weight_decay = getattr(self.config, 'weight_decay', 0.01)
        
        # Scale learning rate for distributed training
        if hasattr(self.config, 'gradient_accumulation_steps'):
            effective_batch_size = (self.config.batch_size_training * 
                                  self.world_size * 
                                  self.config.gradient_accumulation_steps)
            base_lr = learning_rate
            scaled_lr = base_lr * effective_batch_size / (self.config.batch_size_training * self.world_size)
            learning_rate = scaled_lr
            
            if self.rank == 0:
                print(f"Base LR: {base_lr}, Scaled LR: {scaled_lr}, "
                      f"Effective batch size: {effective_batch_size}")
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        return optimizer
    
    def train_epoch(self, 
                   train_dataloader: DataLoader,
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Setup progress bar
        if self.rank == 0:
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"CoT Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device and filter out internal parameters
            # Handle device placement more carefully for CPU/CUDA compatibility
            device = next(self.model.parameters()).device  # Get model's device
            batch = {
                key: batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                for key in batch.keys() if key not in ["idx", "_num_patches_list"]
            }
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                
                if self.rank == 0:
                    pbar.update(1)
            
            # Accumulate loss
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            self.total_train_steps += 1
            
            # Logging
            if self.wandb_run and self.rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": self.total_train_steps,
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }
                self.wandb_run.log(log_dict)
            
            # Update progress bar
            if self.rank == 0:
                current_loss = loss.item() * gradient_accumulation_steps
                pbar.set_description(
                    f"CoT Training Epoch: {epoch+1}/{self.config.num_epochs}, "
                    f"batch {step+1}/{len(train_dataloader)} "
                    f"(loss: {current_loss:.4f})"
                )
        
        if self.rank == 0:
            pbar.close()
        
        # Synchronize across processes
        if self.world_size > 1:
            dist.barrier()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def validate(self, val_dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_dataloader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                # Move batch to device and filter out internal parameters
                # Handle device placement more carefully for CPU/CUDA compatibility
                device = next(self.model.parameters()).device  # Get model's device
                batch = {
                    key: batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                    for key in batch.keys() if key not in ["idx", "_num_patches_list"]
                }
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Accumulate loss across processes
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Log validation metrics
        if self.wandb_run and self.rank == 0:
            log_dict = {
                "eval/loss": avg_loss,
                "eval/epoch": epoch + 1
            }
            self.wandb_run.log(log_dict)
        
        if self.rank == 0:
            print(f"Validation loss: {avg_loss:.4f}")
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Training/validation metrics
        """
        if self.rank != 0:
            return
        
        checkpoint_path = self.save_dir / f"checkpoint_{epoch + 1}"
        
        # Get model state dict
        if isinstance(self.model, (FSDP, DDP)):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        # Save checkpoint
        torch.save(state_dict, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if validation loss improved
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = self.save_dir / "best_cot_model"
            torch.save(state_dict, best_path)
            print(f"Saved best model: {best_path}")
    
    def train(self,
              train_data_path: str,
              val_data_path: str,
              image_root: str,
              start_epoch: int = 0) -> Dict[str, Any]:
        """
        Main training loop for multimodal CoT pre-training.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            image_root: Root directory for images
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            Training summary dictionary
        """
        if self.rank == 0:
            print("=" * 60)
            print("MULTIMODAL COT PRE-TRAINING (STAGE 0)")
            print("=" * 60)
            print(f"Model: {self.config.model_id}")
            print(f"Training samples: Loading...")
            print(f"Validation samples: Loading...")
            print(f"Epochs: {self.config.num_epochs}")
            print(f"Batch size: {self.config.batch_size_training}")
            print(f"Learning rate: {getattr(self.config, 'learning_rate', 1e-5)}")
            print("=" * 60)
        
        # Prepare datasets
        train_dataloader, val_dataloader, val_gen_dataloader = self.prepare_datasets(
            train_data_path, val_data_path, image_root
        )
        
        # Setup optimizer
        optimizer = self.setup_optimizer()
        
        # Training loop
        training_history = []
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
                print("-" * 40)
            
            # Training
            train_metrics = self.train_epoch(train_dataloader, optimizer, epoch)
            
            # Validation
            val_metrics = self.validate(val_dataloader, epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            training_history.append(epoch_metrics)
            
            # Save checkpoint
            save_every = getattr(self.config, 'save_every_n_epochs', 5)
            if (epoch + 1) % save_every == 0 or epoch == self.config.num_epochs - 1:
                self.save_checkpoint(epoch, epoch_metrics)
            
            # Clean up memory
            if self.world_size > 1:
                dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()
            
            if self.rank == 0:
                print(f"Epoch {epoch + 1} completed - "
                      f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("MULTIMODAL COT PRE-TRAINING COMPLETED")
            print("=" * 60)
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Total training steps: {self.total_train_steps}")
            print(f"Checkpoints saved in: {self.save_dir}")
        
        return {
            'training_history': training_history,
            'best_val_loss': self.best_val_loss,
            'total_steps': self.total_train_steps,
            'save_dir': str(self.save_dir)
        }


def create_multimodal_cot_trainer(model: nn.Module,
                                 tokenizer,
                                 config: Config,
                                 rank: int = 0,
                                 world_size: int = 1,
                                 wandb_run=None) -> MultimodalCoTTrainer:
    """
    Factory function to create a MultimodalCoTTrainer.
    
    Args:
        model: Multimodal CoCoNuT model
        tokenizer: Tokenizer with special tokens
        config: Training configuration
        rank: Process rank
        world_size: Total processes
        wandb_run: Weights & Biases run
        
    Returns:
        Configured MultimodalCoTTrainer instance
    """
    return MultimodalCoTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=rank,
        world_size=world_size,
        wandb_run=wandb_run
    )