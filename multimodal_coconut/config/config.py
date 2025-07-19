"""
Configuration management for Multimodal CoCoNuT

Following the original CoCoNuT's simple and elegant configuration approach.
"""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging


class ConfigError(Exception):
    """Custom exception for configuration errors"""
    pass


class Config:
    """
    Simple configuration class that mirrors the original CoCoNuT approach.
    Allows accessing dictionary values as object attributes with enhanced validation.
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
    
    def has(self, key: str) -> bool:
        """Check if configuration has a key"""
        return hasattr(self, key)
    
    def merge(self, other_config: 'Config') -> 'Config':
        """Merge with another config, other_config takes precedence"""
        merged_dict = self.to_dict()
        merged_dict.update(other_config.to_dict())
        return Config(merged_dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


def load_config(config_path: str, validate: bool = True) -> Config:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        validate: Whether to validate the configuration after loading
        
    Returns:
        Config object with loaded settings
        
    Raises:
        ConfigError: If configuration loading or validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    
    try:
        # Read config file
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        # Replace environment variables (${VAR_NAME} format)
        config_str = substitute_env_vars(config_str)
        
        # Parse YAML
        config_dict = yaml.safe_load(config_str)
        
        if config_dict is None:
            raise ConfigError(f"Empty or invalid YAML file: {config_path}")
        
        config = Config(config_dict)
        
        # Validate configuration if requested
        if validate:
            validate_config(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config file {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading config from {config_path}: {e}")


def substitute_env_vars(config_str: str) -> str:
    """
    Substitute environment variables in configuration string.
    Supports both ${VAR_NAME} and ${VAR_NAME:default_value} formats.
    
    Args:
        config_str: Configuration string with environment variable placeholders
        
    Returns:
        Configuration string with environment variables substituted
    """
    # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
    
    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ''
        return os.environ.get(var_name, default_value)
    
    return re.sub(pattern, replace_var, config_str)


def load_config_with_inheritance(config_path: str, base_configs: Optional[List[str]] = None) -> Config:
    """
    Load configuration with inheritance from base configurations.
    
    Args:
        config_path: Path to main configuration file
        base_configs: List of base configuration files to inherit from
        
    Returns:
        Config object with merged settings
    """
    # Start with empty config
    merged_config = Config({})
    
    # Load base configurations first
    if base_configs:
        for base_path in base_configs:
            base_config = load_config(base_path, validate=False)
            merged_config = merged_config.merge(base_config)
    
    # Load main configuration
    main_config = load_config(config_path, validate=False)
    
    # Check if main config specifies a base_config
    if hasattr(main_config, 'base_config'):
        base_path = main_config.base_config
        # Remove base_config from main config to avoid conflicts
        main_dict = main_config.to_dict()
        del main_dict['base_config']
        main_config = Config(main_dict)
        
        # Load and merge base config
        base_config = load_config(base_path, validate=False)
        merged_config = merged_config.merge(base_config)
    
    # Merge main configuration (takes precedence)
    merged_config = merged_config.merge(main_config)
    
    # Validate final merged configuration
    validate_config(merged_config)
    
    return merged_config


def load_config_auto_inherit(config_path: str) -> Config:
    """
    Load configuration with automatic inheritance detection.
    If the config file contains a 'base_config' field, it will be loaded first.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object with inherited settings
    """
    return load_config_with_inheritance(config_path)


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters with comprehensive error checking.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ConfigError: If configuration is invalid
    """
    errors = []
    warnings = []
    
    # Required fields
    required_fields = ['model_id', 'c_thought']
    for field in required_fields:
        if not hasattr(config, field):
            errors.append(f"Missing required config field: {field}")
    
    # Validate CoCoNuT parameters
    try:
        _validate_coconut_params(config, errors)
    except Exception as e:
        errors.append(f"CoCoNuT parameter validation failed: {e}")
    
    # Validate training parameters
    try:
        _validate_training_params(config, errors)
    except Exception as e:
        errors.append(f"Training parameter validation failed: {e}")
    
    # Validate multimodal parameters
    try:
        _validate_multimodal_params(config, errors)
    except Exception as e:
        errors.append(f"Multimodal parameter validation failed: {e}")
    
    # Validate data paths
    try:
        _validate_data_paths(config, warnings)
    except Exception as e:
        warnings.append(f"Data path validation warning: {e}")
    
    # Validate distributed training settings
    try:
        _validate_distributed_params(config, errors)
    except Exception as e:
        errors.append(f"Distributed training validation failed: {e}")
    
    # Print warnings
    for warning in warnings:
        logging.warning(warning)
    
    # Raise errors if any
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ConfigError(error_msg)


def _validate_coconut_params(config: Config, errors: List[str]) -> None:
    """Validate CoCoNuT-specific parameters"""
    # c_thought validation
    c_thought = getattr(config, 'c_thought', None)
    if c_thought is not None:
        if not isinstance(c_thought, int):
            errors.append(f"c_thought must be an integer, got {type(c_thought).__name__}")
        elif c_thought < 1:
            errors.append("c_thought must be >= 1")
    
    # max_latent_stage validation
    if hasattr(config, 'max_latent_stage'):
        max_latent_stage = getattr(config, 'max_latent_stage')
        try:
            max_latent_stage_int = int(max_latent_stage)
            if max_latent_stage_int < 1:
                errors.append("max_latent_stage must be >= 1")
        except (ValueError, TypeError):
            errors.append(f"max_latent_stage must be an integer, got {type(max_latent_stage).__name__}")
    
    # epochs_per_stage validation
    if hasattr(config, 'epochs_per_stage'):
        epochs_per_stage = getattr(config, 'epochs_per_stage')
        try:
            epochs_per_stage_int = int(epochs_per_stage)
            if epochs_per_stage_int < 1:
                errors.append("epochs_per_stage must be >= 1")
        except (ValueError, TypeError):
            errors.append(f"epochs_per_stage must be an integer, got {type(epochs_per_stage).__name__}")
    
    # uniform_prob validation
    if hasattr(config, 'uniform_prob'):
        uniform_prob = getattr(config, 'uniform_prob')
        try:
            uniform_prob_float = float(uniform_prob)
            if not (0.0 <= uniform_prob_float <= 1.0):
                errors.append("uniform_prob must be between 0.0 and 1.0")
        except (ValueError, TypeError):
            errors.append(f"uniform_prob must be a number, got {type(uniform_prob).__name__}")


def _validate_training_params(config: Config, errors: List[str]) -> None:
    """Validate training-specific parameters"""
    # Batch size validation
    for batch_param in ['batch_size_training', 'batch_size_eval']:
        if hasattr(config, batch_param):
            batch_size = getattr(config, batch_param)
            try:
                batch_size_int = int(batch_size)
                if batch_size_int <= 0:
                    errors.append(f"{batch_param} must be positive")
            except (ValueError, TypeError):
                errors.append(f"{batch_param} must be an integer, got {type(batch_size).__name__}")
    
    # Learning rate validation
    if hasattr(config, 'learning_rate'):
        lr = getattr(config, 'learning_rate')
        try:
            lr_float = float(lr)
            if lr_float <= 0:
                errors.append("learning_rate must be positive")
        except (ValueError, TypeError):
            errors.append(f"learning_rate must be a number, got {type(lr).__name__}")
    
    # Weight decay validation
    if hasattr(config, 'weight_decay'):
        wd = getattr(config, 'weight_decay')
        try:
            wd_float = float(wd)
            if wd_float < 0:
                errors.append("weight_decay must be non-negative")
        except (ValueError, TypeError):
            errors.append(f"weight_decay must be a number, got {type(wd).__name__}")
    
    # Epoch validation
    if hasattr(config, 'num_epochs'):
        num_epochs = getattr(config, 'num_epochs')
        try:
            num_epochs_int = int(num_epochs)
            if num_epochs_int <= 0:
                errors.append("num_epochs must be positive")
        except (ValueError, TypeError):
            errors.append(f"num_epochs must be an integer, got {type(num_epochs).__name__}")


def _validate_multimodal_params(config: Config, errors: List[str]) -> None:
    """Validate multimodal-specific parameters"""
    # Image size validation
    if hasattr(config, 'image_size'):
        image_size = getattr(config, 'image_size')
        try:
            image_size_int = int(image_size)
            if image_size_int <= 0:
                errors.append("image_size must be positive")
        except (ValueError, TypeError):
            errors.append(f"image_size must be an integer, got {type(image_size).__name__}")
    
    # Max patches validation
    if hasattr(config, 'max_num_patches'):
        max_patches = getattr(config, 'max_num_patches')
        try:
            max_patches_int = int(max_patches)
            if max_patches_int <= 0:
                errors.append("max_num_patches must be positive")
        except (ValueError, TypeError):
            errors.append(f"max_num_patches must be an integer, got {type(max_patches).__name__}")


def _validate_data_paths(config: Config, warnings: List[str]) -> None:
    """Validate data paths and issue warnings for missing files"""
    data_paths = ['train_data_path', 'val_data_path', 'test_data_path', 'image_root']
    for path_attr in data_paths:
        if hasattr(config, path_attr):
            path_value = getattr(config, path_attr)
            if path_value and path_value != "None":
                path_obj = Path(path_value)
                if not path_obj.exists():
                    warnings.append(f"Data path does not exist: {path_value}")


def _validate_distributed_params(config: Config, errors: List[str]) -> None:
    """Validate distributed training parameters"""
    # Check that only one distributed strategy is enabled
    use_fsdp = getattr(config, 'use_fsdp', False)
    use_ddp = getattr(config, 'use_ddp', False)
    
    if use_fsdp and use_ddp:
        errors.append("Cannot use both FSDP and DDP simultaneously. Choose one.")
    
    # Validate num_workers if specified
    if hasattr(config, 'num_workers'):
        num_workers = getattr(config, 'num_workers')
        try:
            num_workers_int = int(num_workers)
            if num_workers_int < 0:
                errors.append("num_workers must be non-negative")
        except (ValueError, TypeError):
            errors.append(f"num_workers must be an integer, got {type(num_workers).__name__}")


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
        "torch_dtype": "bfloat16",
        
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


def create_config_from_template(template_name: str, **overrides) -> Config:
    """
    Create configuration from a predefined template with optional overrides.
    
    Args:
        template_name: Name of the template ('default', 'cot', 'coconut', 'eval')
        **overrides: Configuration values to override
        
    Returns:
        Config object with template settings and overrides applied
    """
    templates = {
        'default': create_default_config(),
        'cot': _create_cot_template(),
        'coconut': _create_coconut_template(),
        'eval': _create_eval_template(),
        'debug': _create_debug_template(),
    }
    
    if template_name not in templates:
        raise ConfigError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
    
    config_dict = templates[template_name].copy()
    config_dict.update(overrides)
    
    return Config(config_dict)


def _create_cot_template() -> Dict[str, Any]:
    """Create CoT pre-training template"""
    config = create_default_config()
    config.update({
        "name": "multimodal-cot-aokvqa",
        "cot": True,
        "coconut": False,
        "num_epochs": 20,
        "batch_size_training": 4,
        "batch_size_eval": 8,
        "uniform_prob": 0.0,
        "eval_every_n_epochs": 2,
        "save_every_n_epochs": 2,
    })
    return config


def _create_coconut_template() -> Dict[str, Any]:
    """Create CoCoNuT training template"""
    config = create_default_config()
    config.update({
        "name": "multimodal-coconut-aokvqa",
        "cot": False,
        "coconut": True,
        "c_thought": 2,
        "max_latent_stage": 4,
        "epochs_per_stage": 5,
        "uniform_prob": 0.1,
    })
    return config


def _create_eval_template() -> Dict[str, Any]:
    """Create evaluation template"""
    config = create_default_config()
    config.update({
        "name": "multimodal-coconut-eval",
        "only_eval": True,
        "batch_size_eval": 32,
        "num_epochs": 1,
    })
    return config


def _create_debug_template() -> Dict[str, Any]:
    """Create debug template with small settings"""
    config = create_default_config()
    config.update({
        "name": "multimodal-coconut-debug",
        "num_epochs": 2,
        "batch_size_training": 2,
        "batch_size_eval": 2,
        "max_train_samples": 100,
        "max_val_samples": 50,
        "eval_every_n_epochs": 1,
        "save_every_n_epochs": 1,
        "debug": True,
    })
    return config


def print_config_summary(config: Config) -> None:
    """
    Print a formatted summary of the configuration.
    
    Args:
        config: Configuration object to summarize
    """
    print("=" * 60)
    print(f"Configuration Summary: {config.get('name', 'Unnamed')}")
    print("=" * 60)
    
    # Training mode
    mode = "CoT Pre-training" if config.get('cot', False) else "CoCoNuT Training"
    if config.get('only_eval', False):
        mode = "Evaluation Only"
    print(f"Mode: {mode}")
    
    # Model info
    print(f"Model: {config.get('model_id', 'Unknown')}")
    if config.get('load_model_path', 'None') != 'None':
        print(f"Load from: {config.get('load_model_path')}")
    
    # CoCoNuT parameters
    if config.get('coconut', False):
        print(f"CoCoNuT Parameters:")
        print(f"  - c_thought: {config.get('c_thought', 'N/A')}")
        print(f"  - max_latent_stage: {config.get('max_latent_stage', 'N/A')}")
        print(f"  - epochs_per_stage: {config.get('epochs_per_stage', 'N/A')}")
        print(f"  - uniform_prob: {config.get('uniform_prob', 'N/A')}")
    
    # Training parameters
    print(f"Training Parameters:")
    print(f"  - num_epochs: {config.get('num_epochs', 'N/A')}")
    print(f"  - batch_size_training: {config.get('batch_size_training', 'N/A')}")
    print(f"  - learning_rate: {config.get('learning_rate', 'N/A')}")
    
    # Multimodal parameters
    print(f"Multimodal Parameters:")
    print(f"  - image_size: {config.get('image_size', 'N/A')}")
    print(f"  - max_num_patches: {config.get('max_num_patches', 'N/A')}")
    print(f"  - dynamic_preprocess: {config.get('dynamic_preprocess', 'N/A')}")
    
    # Distributed training
    distributed = []
    if config.get('use_fsdp', False):
        distributed.append("FSDP")
    if config.get('use_ddp', False):
        distributed.append("DDP")
    print(f"Distributed: {', '.join(distributed) if distributed else 'None'}")
    
    print("=" * 60)


def get_config_diff(config1: Config, config2: Config) -> Dict[str, tuple]:
    """
    Get differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary with keys that differ and their values as (config1_value, config2_value)
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    differences = {}
    
    for key in all_keys:
        val1 = dict1.get(key, '<missing>')
        val2 = dict2.get(key, '<missing>')
        
        if val1 != val2:
            differences[key] = (val1, val2)
    
    return differences