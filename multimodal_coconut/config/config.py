"""
Configuration management for Multimodal CoCoNuT

Following the original CoCoNuT's simple and elegant configuration approach.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """
    Simple configuration class that mirrors the original CoCoNuT approach.
    Allows accessing dictionary values as object attributes.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        # Set all dictionary items as attributes
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"
    
    def to_dict(self):
        """Convert config back to dictionary"""
        return self.__dict__.copy()
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Get configuration value with default"""
        return getattr(self, key, default)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Read config file
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Replace environment variables (${VAR_NAME} format)
    for key, value in os.environ.items():
        config_str = config_str.replace(f'${{{key}}}', value)
    
    # Parse YAML
    config_dict = yaml.safe_load(config_str)
    
    return Config(config_dict)


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Required fields
    required_fields = ['model_id', 'c_thought']
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate CoCoNuT parameters with type checking
    c_thought = getattr(config, 'c_thought', None)
    if c_thought is not None:
        try:
            c_thought_int = int(c_thought)
            if c_thought_int < 1:
                raise ValueError("c_thought must be >= 1")
        except (ValueError, TypeError):
            raise ValueError(f"c_thought must be an integer, got {type(c_thought)}")
    
    # Validate max_latent_stage
    if hasattr(config, 'max_latent_stage'):
        max_latent_stage = getattr(config, 'max_latent_stage')
        try:
            max_latent_stage_int = int(max_latent_stage)
            if max_latent_stage_int < 1:
                raise ValueError("max_latent_stage must be >= 1")
        except (ValueError, TypeError):
            raise ValueError(f"max_latent_stage must be an integer, got {type(max_latent_stage)}")
    
    # Validate training parameters
    if hasattr(config, 'batch_size_training'):
        batch_size = getattr(config, 'batch_size_training')
        try:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError("batch_size_training must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"batch_size_training must be an integer, got {type(batch_size)}")
    
    if hasattr(config, 'learning_rate'):
        lr = getattr(config, 'learning_rate')
        try:
            lr_float = float(lr)
            if lr_float <= 0:
                raise ValueError("learning_rate must be positive")
        except (ValueError, TypeError):
            raise ValueError(f"learning_rate must be a number, got {type(lr)}")
    
    # Validate data paths if specified
    data_paths = ['train_data_path', 'val_data_path', 'test_data_path']
    for path_attr in data_paths:
        if hasattr(config, path_attr):
            path_value = getattr(config, path_attr)
            if path_value and path_value != "None" and not Path(path_value).exists():
                print(f"Warning: Data path does not exist: {path_value}")


def update_config_for_stage(config: Config, stage: int) -> Config:
    """
    Update configuration based on training stage.
    
    Args:
        config: Base configuration
        stage: Current training stage (0 = CoT, >0 = CoCoNuT)
        
    Returns:
        Updated configuration
    """
    config_copy = Config(config.to_dict())
    
    if stage == 0:  # CoT pre-training
        config_copy.coconut = False
        config_copy.cot = True
    else:  # CoCoNuT training
        config_copy.coconut = True
        config_copy.cot = False
    
    return config_copy


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration dictionary following CoCoNuT patterns.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Experiment settings
        "name": "multimodal-coconut-aokvqa",
        "seed": 42,
        
        # Model configuration
        "model_id": "OpenGVLab/InternVL3-1B-Pretrained",
        "load_model_path": "None",
        "coconut": True,
        "cot": False,
        
        # CoCoNuT parameters
        "c_thought": 2,
        "max_latent_stage": 4,
        "epochs_per_stage": 5,
        "uniform_prob": 0.1,
        "pad_latent_to_max": False,
        "no_cot": False,
        
        # Training parameters
        "num_epochs": 40,
        "batch_size_training": 8,
        "batch_size_eval": 16,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        
        # Multimodal parameters
        "image_size": 448,
        "max_num_patches": 12,
        "use_thumbnail": True,
        "dynamic_preprocess": True,
        
        # Data paths
        "train_data_path": "data/aokvqa/train.json",
        "val_data_path": "data/aokvqa/val.json", 
        "test_data_path": "data/aokvqa/test.json",
        "image_root": "data/aokvqa/images",
        
        # Distributed training
        "use_fsdp": True,
        "use_ddp": False,
        
        # Evaluation
        "only_eval": False,
        "eval_every_n_epochs": 5,
        
        # Checkpointing
        "save_path": "checkpoints",
        "resume": 0,
        "save_every_n_epochs": 5,
    }