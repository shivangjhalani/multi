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

__all__ = [
    'StageManager',
    'StageInfo', 
    'create_stage_manager',
    'get_scheduled_stage',
    'get_stage_data_params'
]