"""
Checkpoint management system for Multimodal CoCoNuT

Handles model state saving and loading with training state preservation,
including stage progression and recovery mechanisms.
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import is_main_process, barrier, get_rank


class CheckpointManager:
    """
    Comprehensive checkpoint management for multimodal CoCoNuT training.
    
    Features:
    - Model state saving and loading for multimodal CoCoNuT
    - Training state preservation including stage progression
    - Checkpoint validation and recovery mechanisms
    - Support for FSDP and DDP distributed training
    - Automatic backup and cleanup
    """
    
    def __init__(self, 
                 save_dir: Union[str, Path],
                 max_checkpoints: int = 5,
                 save_optimizer: bool = True,
                 save_scheduler: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Create save directory
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint tracking
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_metric = None
        
        # Load existing checkpoint history
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load checkpoint history from disk."""
        history_file = self.save_dir / "checkpoint_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoint_history = data.get('checkpoints', [])
                    self.best_checkpoint = data.get('best_checkpoint')
                    self.best_metric = data.get('best_metric')
            except Exception as e:
                print(f"Warning: Could not load checkpoint history: {e}")
    
    def _save_checkpoint_history(self):
        """Save checkpoint history to disk."""
        if not is_main_process():
            return
        
        history_file = self.save_dir / "checkpoint_history.json"
        data = {
            'checkpoints': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'best_metric': self.best_metric
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save checkpoint history: {e}")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       epoch: int,
                       step: int,
                       metrics: Dict[str, float],
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       stage_info: Optional[Dict[str, Any]] = None,
                       config: Optional[Any] = None,
                       tokenizer: Optional[Any] = None,
                       is_best: bool = False,
                       checkpoint_name: Optional[str] = None) -> str:
        """
        Save a comprehensive checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            step: Current training step
            metrics: Training/validation metrics
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            stage_info: CoCoNuT stage information (optional)
            config: Training configuration (optional)
            tokenizer: Tokenizer (optional)
            is_best: Whether this is the best checkpoint
            checkpoint_name: Custom checkpoint name (optional)
            
        Returns:
            Path to saved checkpoint
        """
        if not is_main_process():
            # Wait for main process to save
            barrier()
            return ""
        
        # Generate checkpoint name
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}"
        
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'model_state_dict': self._get_model_state_dict(model),
            'stage_info': stage_info or {},
            'timestamp': torch.tensor(0).item(),  # Placeholder for timestamp
        }
        
        # Add optimizer state
        if optimizer is not None and self.save_optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None and self.save_scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add configuration
        if config is not None:
            if hasattr(config, 'to_dict'):
                checkpoint_data['config'] = config.to_dict()
            else:
                checkpoint_data['config'] = dict(config) if hasattr(config, '__dict__') else str(config)
        
        # Add tokenizer
        if tokenizer is not None:
            tokenizer_path = checkpoint_path.parent / f"{checkpoint_name}_tokenizer"
            try:
                tokenizer.save_pretrained(tokenizer_path)
                checkpoint_data['tokenizer_path'] = str(tokenizer_path)
            except Exception as e:
                print(f"Warning: Could not save tokenizer: {e}")
        
        try:
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update checkpoint history
            checkpoint_info = {
                'name': checkpoint_name,
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'timestamp': checkpoint_data['timestamp']
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # Update best checkpoint if needed
            if is_best or self._is_better_checkpoint(metrics):
                self.best_checkpoint = checkpoint_info
                self.best_metric = metrics
                
                # Create symlink to best checkpoint
                best_link = self.save_dir / "best_checkpoint"
                if best_link.exists() or best_link.is_symlink():
                    best_link.unlink()
                best_link.symlink_to(checkpoint_path.name)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save checkpoint history
            self._save_checkpoint_history()
            
            print(f"Saved checkpoint: {checkpoint_path}")
            if is_best:
                print(f"New best checkpoint with metrics: {metrics}")
            
            # Synchronize with other processes
            barrier()
            
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            barrier()
            raise e
    
    def load_checkpoint(self,
                       checkpoint_path: Union[str, Path],
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       strict: bool = True,
                       load_optimizer: bool = True,
                       load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            strict: Whether to strictly enforce state dict matching
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            model_state_dict = checkpoint_data['model_state_dict']
            self._load_model_state_dict(model, model_state_dict, strict=strict)
            
            # Load optimizer state
            if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint_data:
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    print("Loaded optimizer state")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {e}")
            
            # Load scheduler state
            if scheduler is not None and load_scheduler and 'scheduler_state_dict' in checkpoint_data:
                try:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                    print("Loaded scheduler state")
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
            
            # Extract checkpoint information
            checkpoint_info = {
                'epoch': checkpoint_data.get('epoch', 0),
                'step': checkpoint_data.get('step', 0),
                'metrics': checkpoint_data.get('metrics', {}),
                'stage_info': checkpoint_data.get('stage_info', {}),
                'config': checkpoint_data.get('config', {}),
                'tokenizer_path': checkpoint_data.get('tokenizer_path')
            }
            
            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"Epoch: {checkpoint_info['epoch']}, Step: {checkpoint_info['step']}")
            print(f"Metrics: {checkpoint_info['metrics']}")
            
            return checkpoint_info
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
    
    def load_best_checkpoint(self,
                           model: torch.nn.Module,
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           scheduler: Optional[Any] = None,
                           **kwargs) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            **kwargs: Additional arguments for load_checkpoint
            
        Returns:
            Checkpoint information or None if no best checkpoint exists
        """
        best_link = self.save_dir / "best_checkpoint"
        
        if best_link.exists():
            return self.load_checkpoint(best_link, model, optimizer, scheduler, **kwargs)
        elif self.best_checkpoint:
            return self.load_checkpoint(self.best_checkpoint['path'], model, optimizer, scheduler, **kwargs)
        else:
            print("No best checkpoint found")
            return None
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        return self.checkpoint_history.copy()
    
    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            return False
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Check required fields
            required_fields = ['epoch', 'step', 'model_state_dict']
            for field in required_fields:
                if field not in checkpoint_data:
                    print(f"Missing required field: {field}")
                    return False
            
            # Check model state dict
            model_state = checkpoint_data['model_state_dict']
            if not isinstance(model_state, dict):
                print("Invalid model state dict")
                return False
            
            return True
            
        except Exception as e:
            print(f"Checkpoint validation failed: {e}")
            return False
    
    def _get_model_state_dict(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get model state dict, handling distributed models."""
        if isinstance(model, (FSDP, DDP)):
            return model.module.state_dict()
        else:
            return model.state_dict()
    
    def _load_model_state_dict(self, 
                              model: torch.nn.Module, 
                              state_dict: Dict[str, Any], 
                              strict: bool = True):
        """Load model state dict, handling distributed models."""
        if isinstance(model, (FSDP, DDP)):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)
    
    def _is_better_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics represent a better checkpoint."""
        if self.best_metric is None:
            return True
        
        # Default comparison: lower validation loss is better
        current_val_loss = metrics.get('val_loss', float('inf'))
        best_val_loss = self.best_metric.get('val_loss', float('inf'))
        
        return current_val_loss < best_val_loss
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by timestamp/epoch
        sorted_checkpoints = sorted(
            self.checkpoint_history, 
            key=lambda x: (x.get('epoch', 0), x.get('step', 0))
        )
        
        # Remove oldest checkpoints
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                # Remove tokenizer directory if exists
                tokenizer_path = checkpoint_path.parent / f"{checkpoint_path.stem}_tokenizer"
                if tokenizer_path.exists():
                    shutil.rmtree(tokenizer_path)
                
                self.checkpoint_history.remove(checkpoint_info)
                print(f"Removed old checkpoint: {checkpoint_path}")
                
            except Exception as e:
                print(f"Warning: Could not remove checkpoint {checkpoint_info['path']}: {e}")


def create_checkpoint_manager(save_dir: Union[str, Path], **kwargs) -> CheckpointManager:
    """
    Factory function to create a CheckpointManager.
    
    Args:
        save_dir: Directory to save checkpoints
        **kwargs: Additional arguments for CheckpointManager
        
    Returns:
        Configured CheckpointManager instance
    """
    return CheckpointManager(save_dir, **kwargs)


def auto_resume_from_checkpoint(checkpoint_dir: Union[str, Path],
                               model: torch.nn.Module,
                               optimizer: Optional[torch.optim.Optimizer] = None,
                               scheduler: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Automatically resume from the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        
    Returns:
        Checkpoint information or None if no checkpoint found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint manager
    manager = CheckpointManager(checkpoint_dir)
    
    # Try to load best checkpoint first
    checkpoint_info = manager.load_best_checkpoint(model, optimizer, scheduler)
    
    if checkpoint_info is None:
        # Try to find any checkpoint
        checkpoints = manager.list_checkpoints()
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: (x.get('epoch', 0), x.get('step', 0)))
            checkpoint_info = manager.load_checkpoint(
                latest_checkpoint['path'], model, optimizer, scheduler
            )
    
    return checkpoint_info