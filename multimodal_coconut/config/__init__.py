"""
Configuration management for Multimodal CoCoNuT
"""

from .config import (
    Config,
    ConfigError,
    load_config,
    load_config_with_inheritance,
    substitute_env_vars,
    validate_config,
    update_config_for_stage,
    create_default_config,
    create_config_from_template,
    print_config_summary,
    get_config_diff
)

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "load_config_with_inheritance",
    "substitute_env_vars",
    "validate_config", 
    "update_config_for_stage",
    "create_default_config",
    "create_config_from_template",
    "print_config_summary",
    "get_config_diff"
]