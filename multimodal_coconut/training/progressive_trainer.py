"""
Progressive Training Orchestrator for Multimodal CoCoNuT

This module implements the complete training pipeline that orchestrates the full
CoCoNuT curriculum from Stage 0 (CoT pre-training) through progressive latent stages.

Key features:
- Complete curriculum orchestration (Stage 0 → Stage 1+ progression)
- Automatic stage transitions based on epoch and epochs_per_stage
- Seamless integration between CoT and CoCoNuT trainers
- Checkpoint management across stage transitions
- Following original CoCoNuT methodology exactly
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..config import Config
from .stage_manager import StageManager
from .multimodal_cot_trainer import create_multimodal_cot_trainer
from .multimodal_coconut_trainer import create_multimodal_coconut_trainer


class ProgressiveTrainingOrchestrator:
    """
    Orchestrates the complete multimodal CoCoNuT training curriculum.
    
    This class manages the transition from CoT pre-training (Stage 0) to
    progressive CoCoNuT training (Stages 1+), following the original CoCoNuT
    methodology exactly.
    
    Training flow:
    1. Stage 0: CoT pre-training (if enabled)
    2. Stages 1+: Progressive latent token replacement
    3. Automatic stage transitions based on epochs_per_stage
    4. Checkpoint management and resuming
    """
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 config: Config,
                 rank: int = 0,
                 world_size: int = 1,
                 wandb_run=None):
        """
        Initialize progressive training orchestrator
        
        Args:
            model: Multimodal CoCoNuT model
            tokenizer: Tokenizer with special tokens
            config: Training configuration
            rank: Process rank for distributed training
            world_size: Total number of processes
            wandb_run: Weights & Biases run for logging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.wandb_run = wandb_run
        
        # Stage manager for curriculum
        self.stage_manager = StageManager(config)
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration for progressive training"""
        required_params = [
            'num_epochs', 'epochs_per_stage', 'max_latent_stage', 'c_thought'
        ]
        for param in required_params:
            if not hasattr(self.config, param):
                raise ValueError(f"Missing required config parameter: {param}")
        
        # Ensure we have proper training mode configuration
        if not hasattr(self.config, 'coconut'):
            self.config.coconut = True
        if not hasattr(self.config, 'cot'):
            self.config.cot = False
    
    def determine_training_mode(self, epoch: int) -> str:
        """
        Determine training mode based on current epoch and configuration.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Training mode: 'cot' for Stage 0, 'coconut' for Stages 1+
        """
        scheduled_stage = self.stage_manager.get_current_stage(epoch)
        
        if scheduled_stage == 0:
            return 'cot'
        else:
            return 'coconut'
    
    def create_trainer_for_mode(self, mode: str):
        """
        Create appropriate trainer based on training mode.
        
        Args:
            mode: Training mode ('cot' or 'coconut')
            
        Returns:
            Configured trainer instance
        """
        if mode == 'cot':
            # Create CoT trainer for Stage 0
            cot_config = self.config.copy()
            cot_config.cot = True
            cot_config.coconut = False
            
            return create_multimodal_cot_trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=cot_config,
                rank=self.rank,
                world_size=self.world_size,
                wandb_run=self.wandb_run
            )
        
        elif mode == 'coconut':
            # Create CoCoNuT trainer for Stages 1+
            coconut_config = self.config.copy()
            coconut_config.cot = False
            coconut_config.coconut = True
            
            return create_multimodal_coconut_trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=coconut_config,
                rank=self.rank,
                world_size=self.world_size,
                wandb_run=self.wandb_run
            )
        
        else:
            raise ValueError(f"Unknown training mode: {mode}")
    
    def should_transition_stage(self, epoch: int) -> bool:
        """
        Check if we should transition to a new stage.
        
        Args:
            epoch: Current epoch
            
        Returns:
            True if stage transition is needed
        """
        current_stage = self.stage_manager.get_current_stage(epoch)
        
        if epoch == 0:
            return True  # Always transition at start
        
        previous_stage = self.stage_manager.get_current_stage(epoch - 1)
        return current_stage != previous_stage
    
    def train_progressive(self,
                         train_data_path: str,
                         val_data_path: str,
                         image_root: str,
                         start_epoch: int = 0) -> Dict[str, Any]:
        """
        Execute progressive training with automatic stage transitions.
        
        This implements the core CoCoNuT curriculum:
        - Stage 0: CoT pre-training (if epochs allow)
        - Stages 1+: Progressive latent token replacement
        - Automatic transitions based on epochs_per_stage
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            image_root: Root directory for images
            start_epoch: Starting epoch (for resuming)
            
        Returns:
            Complete training summary
        """
        if self.rank == 0:
            print("=" * 70)
            print("MULTIMODAL COCONUT PROGRESSIVE TRAINING ORCHESTRATOR")
            print("=" * 70)
            print(f"Total epochs: {self.config.num_epochs}")
            print(f"Epochs per stage: {self.config.epochs_per_stage}")
            print(f"Max latent stage: {self.config.max_latent_stage}")
            print("=" * 70)
            
            # Print full curriculum
            self.stage_manager.print_curriculum_summary(self.config.num_epochs)
        
        current_trainer = None
        current_mode = None
        
        # Track training across all stages
        all_training_history = []
        stage_summaries = []
        
        epoch = start_epoch
        while epoch < self.config.num_epochs:
            # Check if we need to transition stages
            if self.should_transition_stage(epoch):
                new_mode = self.determine_training_mode(epoch)
                scheduled_stage = self.stage_manager.get_current_stage(epoch)
                
                if new_mode != current_mode:
                    if self.rank == 0:
                        stage_info = self.stage_manager.get_stage_info(scheduled_stage)
                        print(f"\n🔄 STAGE TRANSITION: {stage_info.description}")
                        print(f"Switching to {new_mode.upper()} training mode")
                        print("-" * 70)
                    
                    # Create new trainer for this mode
                    current_trainer = self.create_trainer_for_mode(new_mode)
                    current_mode = new_mode
            
            # Determine how many epochs to run in current stage
            current_stage = self.stage_manager.get_current_stage(epoch)
            
            # Calculate end epoch for current stage
            if current_stage == 0:
                # Stage 0 runs for epochs_per_stage epochs (or until stage 1 starts)
                stage_end_epoch = min(
                    self.config.epochs_per_stage,
                    self.config.num_epochs
                )
            else:
                # Calculate when next stage starts
                next_stage_start = (current_stage + 1) * self.config.epochs_per_stage
                stage_end_epoch = min(next_stage_start, self.config.num_epochs)
            
            # Run training for this stage
            epochs_in_stage = stage_end_epoch - epoch
            
            if epochs_in_stage > 0:
                if self.rank == 0:
                    print(f"\n🚀 Running {epochs_in_stage} epochs in Stage {current_stage}")
                    print(f"Epochs {epoch + 1} to {epoch + epochs_in_stage}")
                
                # Update trainer's config for this stage
                stage_config = current_trainer.config.copy()
                stage_config.num_epochs = epochs_in_stage
                current_trainer.config = stage_config
                
                # Run training for this stage
                if current_mode == 'cot':
                    stage_results = current_trainer.train(
                        train_data_path=train_data_path,
                        val_data_path=val_data_path,
                        image_root=image_root,
                        start_epoch=0  # Reset for each stage
                    )
                else:  # coconut mode
                    stage_results = current_trainer.train(
                        train_data_path=train_data_path,
                        val_data_path=val_data_path,
                        image_root=image_root,
                        start_epoch=0  # Reset for each stage
                    )
                
                # Record stage summary
                stage_summary = {
                    'stage': current_stage,
                    'mode': current_mode,
                    'start_epoch': epoch + 1,
                    'end_epoch': epoch + epochs_in_stage,
                    'epochs_trained': epochs_in_stage,
                    'best_val_loss': stage_results.get('best_val_loss', float('inf')),
                    'total_steps': stage_results.get('total_steps', 0)
                }
                stage_summaries.append(stage_summary)
                
                # Add stage results to overall history
                if 'training_history' in stage_results:
                    # Adjust epoch numbers to be global
                    for hist_entry in stage_results['training_history']:
                        hist_entry['global_epoch'] = epoch + hist_entry['epoch']
                        hist_entry['stage'] = current_stage
                        hist_entry['mode'] = current_mode
                    all_training_history.extend(stage_results['training_history'])
                
                if self.rank == 0:
                    print(f"✅ Stage {current_stage} completed - "
                          f"Best Val Loss: {stage_results.get('best_val_loss', 'N/A')}")
            
            # Move to next stage
            epoch = stage_end_epoch
        
        # Final summary
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("PROGRESSIVE TRAINING COMPLETED")
            print("=" * 70)
            
            for summary in stage_summaries:
                print(f"Stage {summary['stage']} ({summary['mode'].upper()}): "
                      f"Epochs {summary['start_epoch']}-{summary['end_epoch']}, "
                      f"Best Loss: {summary['best_val_loss']:.4f}")
            
            overall_best_loss = min([s['best_val_loss'] for s in stage_summaries])
            total_steps = sum([s['total_steps'] for s in stage_summaries])
            print(f"\nOverall Best Validation Loss: {overall_best_loss:.4f}")
            print(f"Total Training Steps: {total_steps}")
            print("=" * 70)
        
        return {
            'training_history': all_training_history,
            'stage_summaries': stage_summaries,
            'best_val_loss': min([s['best_val_loss'] for s in stage_summaries]),
            'total_steps': sum([s['total_steps'] for s in stage_summaries]),
            'final_stage': stage_summaries[-1]['stage'] if stage_summaries else 0
        }


def create_progressive_trainer(model: nn.Module,
                              tokenizer,
                              config: Config,
                              rank: int = 0,
                              world_size: int = 1,
                              wandb_run=None) -> ProgressiveTrainingOrchestrator:
    """
    Factory function to create a ProgressiveTrainingOrchestrator.
    
    Args:
        model: Multimodal CoCoNuT model
        tokenizer: Tokenizer with special tokens
        config: Training configuration
        rank: Process rank
        world_size: Total processes
        wandb_run: Weights & Biases run
        
    Returns:
        Configured ProgressiveTrainingOrchestrator instance
    """
    return ProgressiveTrainingOrchestrator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=rank,
        world_size=world_size,
        wandb_run=wandb_run
    )