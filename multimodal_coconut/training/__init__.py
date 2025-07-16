"""
Training module for Multimodal CoCoNuT

This module contains the staged training curriculum system and related utilities
for training multimodal CoCoNuT models.
"""

from .stage_manager import (
    StageManager,
    StageInfo,
    create_stage_manager,
    get_scheduled_stage,
    get_stage_data_params
)

from .multimodal_cot_trainer import (
    MultimodalCoTTrainer,
    create_multimodal_cot_trainer
)

__all__ = [
    'StageManager',
    'StageInfo', 
    'create_stage_manager',
    'get_scheduled_stage',
    'get_stage_data_params',
    'MultimodalCoTTrainer',
    'create_multimodal_cot_trainer'
]