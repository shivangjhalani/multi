"""
Training module for Multimodal CoCoNuT

This module contains all training-related components including:
- Stage management for curriculum learning
- Training loops and optimization
- Validation and evaluation logic
"""

from .stage_manager import StageManager, create_stage_manager, get_scheduled_stage, get_stage_data_params

__all__ = [
    'StageManager',
    'create_stage_manager', 
    'get_scheduled_stage',
    'get_stage_data_params'
]