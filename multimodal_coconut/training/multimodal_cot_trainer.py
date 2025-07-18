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
from ..utils.distributed import (
    get_distributed_sampler,
    synchronize_multimodal_batch,
    reduce_tensor,
    barrier
)
from ..utils.checkpoint import CheckpointManager
from ..utils.memory import MemoryOptimizer, memory_efficient_forward
from ..utils.logging import (
    get_logger,
    log_metrics,
    MetricsTracker,
    ExperimentTracker,
    MultimodalDebugger,
    create_experiment_tracker,
    create_multimodal_debugger
)


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
        
        # Setup logger
        self.logger = get_logger()
        
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
        
        # Initialize experiment tracker
        if rank == 0:
            self.experiment_tracker = create_experiment_tracker(config.to_dict())
            if self.experiment_tracker.run:
                self.experiment_tracker.log_model_architecture(model)
        else:
            self.experiment_tracker = None
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            window_size=getattr(config, 'metrics_window_size', 100)
        )
        
        # Initialize debugger if enabled
        if getattr(config, 'enable_debugging', False):
            self.debugger = create_multimodal_debugger(config.to_dict())
        else:
            self.debugger = None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            max_checkpoints=getattr(config, 'max_checkpoints', 5),
            save_optimizer=getattr(config, 'save_optimizer_state', True),
            save_scheduler=getattr(config, 'save_scheduler_state', True)
        )
        
        # Initialize memory optimizer
        self.memory_optimizer = MemoryOptimizer(
            enable_gradient_checkpointing=getattr(config, 'enable_gradient_checkpointing', True),
            enable_auto_batch_reduction=getattr(config, 'enable_auto_batch_reduction', True),
            min_batch_size=getattr(config, 'min_batch_size', 1),
            memory_cleanup_frequency=getattr(config, 'memory_cleanup_frequency', 100)
        )
        
        # Setup gradient checkpointing
        if self.memory_optimizer.enable_gradient_checkpointing:
            self.model = self.memory_optimizer.setup_gradient_checkpointing(self.model)
        
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
        
        # Create data loaders with distributed samplers
        train_dataloader = DataLoader(
            dataset_train,
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=(self.world_size == 1),  # Only shuffle if not using distributed sampler
            pin_memory=True,
            batch_size=self.config.batch_size_training,
            collate_fn=self.collator,
            sampler=get_distributed_sampler(dataset_train, shuffle=True)
        )
        
        val_loss_dataloader = DataLoader(
            dataset_loss_val,
            num_workers=getattr(self.config, 'num_workers', 1),
            shuffle=False,
            pin_memory=True,
            batch_size=getattr(self.config, 'batch_size_eval', self.config.batch_size_training),
            collate_fn=self.collator,
            sampler=get_distributed_sampler(dataset_loss_val, shuffle=False)
        )
        
        val_gen_dataloader = DataLoader(
            dataset_gen_val,
            num_workers=getattr(self.config, 'num_workers', 1),
            pin_memory=True,
            batch_size=1,  # Generation typically done with batch_size=1
            collate_fn=self.collator,
            sampler=get_distributed_sampler(dataset_gen_val, shuffle=False)
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
            
            # Synchronize multimodal batch across processes for consistent training
            if self.world_size > 1:
                batch = synchronize_multimodal_batch(batch)
            
            # Memory-efficient forward pass with OOM handling
            try:
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.memory_optimizer.enable_auto_batch_reduction:
                    # Handle OOM by reducing batch size
                    def forward_fn(optimized_batch):
                        return self.model(**optimized_batch)
                    
                    outputs, batch = self.memory_optimizer.handle_oom_error(batch, forward_fn)
                    loss = outputs.loss / gradient_accumulation_steps
                    
                    if self.rank == 0:
                        print(f"Recovered from OOM at step {step}")
                else:
                    raise e
            
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
            
            # Memory optimization step
            self.memory_optimizer.step()
            
            # Enhanced logging with experiment tracker
            current_loss = loss.item() * gradient_accumulation_steps
            
            # Update metrics tracker
            self.metrics_tracker.update(
                train_loss=current_loss,
                learning_rate=optimizer.param_groups[0]['lr']
            )
            
            # Log to experiment tracker
            if self.experiment_tracker and self.rank == 0:
                log_dict = {
                    "train/epoch": epoch + 1,
                    "train/step": self.total_train_steps,
                    "train/loss": current_loss,
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }
                
                # Add memory optimization metrics
                memory_stats = self.memory_optimizer.get_memory_stats()
                for key, value in memory_stats.items():
                    if isinstance(value, (int, float)):
                        log_dict[f"memory/{key}"] = value
                
                self.experiment_tracker.log_training_metrics(
                    log_dict, 
                    step=self.total_train_steps,
                    epoch=epoch,
                    stage=0  # CoT pre-training is stage 0
                )
                
                # Log memory usage periodically
                if self.total_train_steps % 100 == 0:
                    self.experiment_tracker.log_memory_usage(step=self.total_train_steps)
            
            # Debug batch and model outputs if debugging is enabled
            if self.debugger and self.rank == 0:
                debug_frequency = getattr(self.config, 'debug_frequency', 1000)
                if self.total_train_steps % debug_frequency == 0:
                    self.debugger.debug_batch(batch, self.tokenizer, step=self.total_train_steps)
                    self.debugger.debug_model_outputs(outputs, step=self.total_train_steps)
                    
                    # Debug gradients after backward pass
                    if (step + 1) % gradient_accumulation_steps == 0:
                        self.debugger.debug_gradients(self.model, step=self.total_train_steps)
                
                # Profile memory usage periodically
                if self.total_train_steps % 500 == 0:
                    self.debugger.profile_memory_usage(step=self.total_train_steps)
            
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
            barrier()
        
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
                
                # Synchronize multimodal batch across processes for consistent validation
                if self.world_size > 1:
                    batch = synchronize_multimodal_batch(batch)
                
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
        
        # Enhanced validation logging with experiment tracker
        if self.experiment_tracker and self.rank == 0:
            val_results = {
                "val_loss": avg_loss,
                "val_num_batches": num_batches
            }
            self.experiment_tracker.log_validation_results(
                val_results, 
                step=self.total_train_steps
            )
        
        # Update metrics tracker
        self.metrics_tracker.update(val_loss=avg_loss)
        
        if self.rank == 0:
            self.logger.info(f"Validation loss: {avg_loss:.4f}")
            
            # Log recent metrics statistics
            recent_stats = self.metrics_tracker.get_statistics()
            if recent_stats:
                self.logger.info("Recent training statistics:")
                for metric, stats in recent_stats.items():
                    self.logger.info(f"  {metric}: mean={stats['mean']:.4f}, "
                                   f"std={stats['std']:.4f}, "
                                   f"min={stats['min']:.4f}, "
                                   f"max={stats['max']:.4f}")
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, 
                       epoch: int, 
                       metrics: Dict[str, float],
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[Any] = None):
        """
        Save model checkpoint using the checkpoint management system.
        
        Args:
            epoch: Current epoch number
            metrics: Training/validation metrics
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
        """
        # Determine if this is the best checkpoint
        is_best = False
        if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            is_best = True
        
        # Prepare stage information for CoT training
        stage_info = {
            'stage': 0,  # CoT pre-training is stage 0
            'stage_type': 'cot',
            'total_train_steps': self.total_train_steps,
            'current_epoch': epoch
        }
        
        # Save checkpoint using checkpoint manager
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            epoch=epoch,
            step=self.total_train_steps,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            stage_info=stage_info,
            config=self.config,
            tokenizer=self.tokenizer,
            is_best=is_best
        )
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to restore state (optional)
            scheduler: Scheduler to restore state (optional)
            
        Returns:
            Checkpoint information dictionary
        """
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Restore training state
        if checkpoint_info:
            self.current_epoch = checkpoint_info.get('epoch', 0)
            self.total_train_steps = checkpoint_info.get('step', 0)
            
            # Restore stage information
            stage_info = checkpoint_info.get('stage_info', {})
            if 'total_train_steps' in stage_info:
                self.total_train_steps = stage_info['total_train_steps']
            
            # Update best validation loss
            metrics = checkpoint_info.get('metrics', {})
            if 'val_loss' in metrics:
                self.best_val_loss = metrics['val_loss']
        
        return checkpoint_info
    
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
                self.save_checkpoint(epoch, epoch_metrics, optimizer=optimizer)
            
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