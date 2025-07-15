"""
Logging utilities for Multimodal CoCoNuT
"""

import logging
import sys
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
from .distributed import is_main_process


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_wandb: bool = True,
    wandb_project: str = "multimodal-coconut",
    wandb_config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_config: W&B configuration dictionary
        
    Returns:
        Configured logger
    """
    # Only setup logging on main process
    if not is_main_process():
        logging.getLogger().setLevel(logging.WARNING)
        return logging.getLogger()
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger("multimodal_coconut")
    
    # Initialize Weights & Biases
    if use_wandb:
        try:
            wandb.init(
                project=wandb_project,
                config=wandb_config or {},
                reinit=True
            )
            logger.info("Initialized Weights & Biases logging")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    return logger


def get_logger(name: str = "multimodal_coconut") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)


def log_config(config: Any, logger: Optional[logging.Logger] = None):
    """
    Log configuration settings.
    
    Args:
        config: Configuration object
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    if not is_main_process():
        return
    
    logger.info("Configuration:")
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = dict(config)
    
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    
    # Log to W&B if available
    if wandb.run is not None:
        wandb.config.update(config_dict)


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "",
    logger: Optional[logging.Logger] = None
):
    """
    Log metrics to console and W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step number
        prefix: Prefix for metric names
        logger: Logger instance
    """
    if logger is None:
        logger = get_logger()
    
    if not is_main_process():
        return
    
    # Log to console
    metric_str = " | ".join([f"{prefix}{k}: {v:.4f}" if isinstance(v, float) else f"{prefix}{k}: {v}" 
                            for k, v in metrics.items()])
    
    if step is not None:
        logger.info(f"Step {step} | {metric_str}")
    else:
        logger.info(metric_str)
    
    # Log to W&B if available
    if wandb.run is not None:
        wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        if step is not None:
            wandb.log(wandb_metrics, step=step)
        else:
            wandb.log(wandb_metrics)


class MetricsTracker:
    """Simple metrics tracking utility."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {key: self.metrics[key] / self.counts[key] 
                for key in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def log(self, step: Optional[int] = None, prefix: str = "", logger: Optional[logging.Logger] = None):
        """Log current averages."""
        averages = self.get_averages()
        if averages:
            log_metrics(averages, step=step, prefix=prefix, logger=logger)