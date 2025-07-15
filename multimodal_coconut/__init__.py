"""
Multimodal CoCoNuT: Chain of Continuous Thought for Multimodal Reasoning

This package extends the CoCoNuT (Chain of Continuous Thought) methodology 
to multimodal reasoning using InternVL3 as the base model.
"""

__version__ = "0.1.0"
__author__ = "Multimodal CoCoNuT Team"

from .config import Config, load_config, validate_config
from .model import MultimodalCoconut
from .utils import (
    set_seed,
    setup_logging,
    get_logger,
    init_distributed_training,
    is_main_process
)

__all__ = [
    "Config",
    "load_config", 
    "validate_config",
    "MultimodalCoconut",
    "set_seed",
    "setup_logging",
    "get_logger",
    "init_distributed_training",
    "is_main_process"
]