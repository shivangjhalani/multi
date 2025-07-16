"""
Stage Management System for Multimodal CoCoNuT

This module implements the staged training curriculum system that gradually transitions
from standard multimodal chain-of-thought to continuous latent reasoning.

Following the original CoCoNuT pattern exactly:
- Stage 0: Standard multimodal CoT (cot=True, coconut=False)
- Stage k: Replace first k reasoning steps with k*c_thought latent tokens
- Progressive deepening up to max_latent_stage

Key features:
- Curriculum progression based on epoch and epochs_per_stage
- Stage-specific data preparation functions
- Uniform probability mixing for multi-stage training
- Configuration management for different training phases
"""

import random
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from ..config import Config


@dataclass
class StageInfo:
    """Information about a specific training stage"""
    stage_number: int
    is_cot_stage: bool  # True for stage 0 (CoT), False for CoCoNuT stages
    num_latent_steps: int  # Number of reasoning steps to replace with latent tokens
    num_latent_tokens: int  # Total number of latent tokens (num_latent_steps * c_thought)
    description: str


class StageManager:
    """
    Manages the staged training curriculum for multimodal CoCoNuT.
    
    This class follows the original CoCoNuT pattern exactly:
    - scheduled_stage = 0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
    - Stage 0 = standard multimodal CoT training
    - Stage k = replace first k reasoning steps with continuous thoughts
    """
    
    def __init__(self, config: Config):
        """
        Initialize stage manager with configuration
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        
        # Core CoCoNuT parameters
        self.epochs_per_stage = getattr(config, 'epochs_per_stage', 5)
        self.max_latent_stage = getattr(config, 'max_latent_stage', 4)
        self.c_thought = getattr(config, 'c_thought', 2)
        self.uniform_prob = getattr(config, 'uniform_prob', 0.1)
        
        # Training mode flags
        self.cot = getattr(config, 'cot', False)
        self.coconut = getattr(config, 'coconut', True)
        self.no_cot = getattr(config, 'no_cot', False)
        
        # Additional parameters
        self.pad_latent_to_max = getattr(config, 'pad_latent_to_max', False)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate stage management configuration"""
        if self.epochs_per_stage <= 0:
            raise ValueError("epochs_per_stage must be positive")
        
        if self.max_latent_stage <= 0:
            raise ValueError("max_latent_stage must be positive")
        
        if self.c_thought <= 0:
            raise ValueError("c_thought must be positive")
        
        if not (0 <= self.uniform_prob <= 1):
            raise ValueError("uniform_prob must be between 0 and 1")
    
    def get_current_stage(self, epoch: int) -> int:
        """
        Calculate current training stage based on epoch.
        
        This follows the original CoCoNuT formula exactly:
        scheduled_stage = 0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Current stage number (0 for CoT, 1+ for CoCoNuT stages)
        """
        if self.cot or self.no_cot:
            # Force stage 0 for pure CoT training or no-CoT mode
            return 0
        
        # Standard CoCoNuT curriculum progression
        return epoch // self.epochs_per_stage
    
    def get_stage_info(self, stage: int) -> StageInfo:
        """
        Get detailed information about a specific stage
        
        Args:
            stage: Stage number
            
        Returns:
            StageInfo object with stage details
        """
        if stage == 0:
            return StageInfo(
                stage_number=0,
                is_cot_stage=True,
                num_latent_steps=0,
                num_latent_tokens=0,
                description="Standard multimodal chain-of-thought training"
            )
        
        # CoCoNuT stages
        effective_stage = min(stage, self.max_latent_stage)
        num_latent_tokens = effective_stage * self.c_thought
        
        return StageInfo(
            stage_number=stage,
            is_cot_stage=False,
            num_latent_steps=effective_stage,
            num_latent_tokens=num_latent_tokens,
            description=f"CoCoNuT stage {stage}: Replace first {effective_stage} steps with {num_latent_tokens} latent tokens"
        )
    
    def should_use_uniform_mixing(self) -> bool:
        """
        Determine if uniform probability mixing should be used for this sample.
        
        Returns:
            True if uniform mixing should be applied
        """
        return random.random() < self.uniform_prob
    
    def sample_random_stage(self, max_available_steps: int) -> int:
        """
        Sample a random stage for uniform probability mixing.
        
        Args:
            max_available_steps: Maximum number of reasoning steps available in the sample
            
        Returns:
            Randomly sampled stage number
        """
        # Sample from 0 to max_available_steps (inclusive)
        # This matches the original CoCoNuT pattern
        return random.choice(list(range(max_available_steps + 1)))
    
    def get_effective_stage_for_sample(self, 
                                     scheduled_stage: int, 
                                     sample_steps: List[str]) -> Tuple[int, int, int]:
        """
        Get the effective stage parameters for a specific sample.
        
        This implements the core logic from the original CoCoNuT:
        - Apply uniform probability mixing if enabled
        - Handle max_latent_stage constraints
        - Calculate n_skip_steps and n_latent_tokens
        
        Args:
            scheduled_stage: Current scheduled stage from curriculum
            sample_steps: List of reasoning steps for this sample
            
        Returns:
            Tuple of (effective_stage, n_skip_steps, n_latent_tokens)
        """
        # Apply uniform probability mixing
        if self.should_use_uniform_mixing():
            effective_stage = self.sample_random_stage(len(sample_steps))
        else:
            effective_stage = scheduled_stage
        
        # Handle max_latent_stage constraint
        if effective_stage > self.max_latent_stage:
            n_skip_steps = 10000  # Skip all steps (original CoCoNuT pattern)
            if self.pad_latent_to_max:
                n_latent_tokens = self.max_latent_stage
            else:
                n_latent_tokens = min(len(sample_steps), self.max_latent_stage)
        else:
            n_skip_steps = effective_stage
            n_latent_tokens = effective_stage
        
        # Handle no_cot mode
        if self.no_cot:
            n_skip_steps = 100  # Skip all steps
            n_latent_tokens = 0
        
        # Apply c_thought multiplier
        n_latent_tokens *= self.c_thought
        
        return effective_stage, n_skip_steps, n_latent_tokens
    
    def update_config_for_stage(self, stage: int) -> Config:
        """
        Update configuration based on current training stage.
        
        Args:
            stage: Current training stage
            
        Returns:
            Updated configuration for the stage
        """
        # Create a copy of the config
        config_dict = self.config.to_dict()
        
        if stage == 0:  # CoT pre-training
            config_dict['coconut'] = False
            config_dict['cot'] = True
        else:  # CoCoNuT training
            config_dict['coconut'] = True
            config_dict['cot'] = False
        
        return Config(config_dict)
    
    def get_training_summary(self, total_epochs: int) -> Dict[str, Any]:
        """
        Get a summary of the training curriculum.
        
        Args:
            total_epochs: Total number of training epochs
            
        Returns:
            Dictionary with curriculum summary
        """
        stages = []
        current_stage = 0
        
        for epoch in range(total_epochs):
            stage = self.get_current_stage(epoch)
            if stage != current_stage:
                current_stage = stage
            
            if not stages or stages[-1]['stage'] != stage:
                stage_info = self.get_stage_info(stage)
                stages.append({
                    'stage': stage,
                    'start_epoch': epoch,
                    'description': stage_info.description,
                    'num_latent_tokens': stage_info.num_latent_tokens
                })
        
        return {
            'total_epochs': total_epochs,
            'epochs_per_stage': self.epochs_per_stage,
            'max_latent_stage': self.max_latent_stage,
            'c_thought': self.c_thought,
            'uniform_prob': self.uniform_prob,
            'stages': stages
        }
    
    def print_curriculum_summary(self, total_epochs: int):
        """
        Print a human-readable summary of the training curriculum.
        
        Args:
            total_epochs: Total number of training epochs
        """
        summary = self.get_training_summary(total_epochs)
        
        print("=" * 60)
        print("MULTIMODAL COCONUT TRAINING CURRICULUM")
        print("=" * 60)
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Epochs per stage: {summary['epochs_per_stage']}")
        print(f"Max latent stage: {summary['max_latent_stage']}")
        print(f"Continuous thoughts per step: {summary['c_thought']}")
        print(f"Uniform mixing probability: {summary['uniform_prob']}")
        print()
        
        print("Stage progression:")
        for stage_info in summary['stages']:
            print(f"  Stage {stage_info['stage']} (epoch {stage_info['start_epoch']}+): {stage_info['description']}")
        
        print("=" * 60)


def create_stage_manager(config: Config) -> StageManager:
    """
    Factory function to create a StageManager instance.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured StageManager instance
    """
    return StageManager(config)


# Utility functions for backward compatibility with original CoCoNuT patterns

def get_scheduled_stage(epoch: int, config: Config) -> int:
    """
    Get scheduled stage for a given epoch (backward compatibility function).
    
    Args:
        epoch: Current epoch
        config: Configuration object
        
    Returns:
        Scheduled stage number
    """
    stage_manager = StageManager(config)
    return stage_manager.get_current_stage(epoch)


def get_stage_data_params(scheduled_stage: int, 
                         sample_steps: List[str], 
                         config: Config) -> Tuple[int, int, int]:
    """
    Get stage-specific data parameters (backward compatibility function).
    
    Args:
        scheduled_stage: Current scheduled stage
        sample_steps: List of reasoning steps
        config: Configuration object
        
    Returns:
        Tuple of (effective_stage, n_skip_steps, n_latent_tokens)
    """
    stage_manager = StageManager(config)
    return stage_manager.get_effective_stage_for_sample(scheduled_stage, sample_steps)