"""
Comprehensive logging utilities for Multimodal CoCoNuT

This module provides:
- Weights & Biases integration for experiment tracking
- Comprehensive logging for training metrics and model performance
- Debugging utilities for multimodal training issues
- Memory usage tracking and performance monitoring
- Visual debugging tools for multimodal data
"""

import logging
import sys
import os
import time
import json
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict, deque
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
    """Enhanced metrics tracking utility with history and statistics."""
    
    def __init__(self, window_size: int = 100):
        self.metrics = {}
        self.counts = {}
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
            self.history[key].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        return {key: self.metrics[key] / self.counts[key] 
                for key in self.metrics.keys()}
    
    def get_recent_averages(self, window: Optional[int] = None) -> Dict[str, float]:
        """Get recent average values within a window."""
        if window is None:
            window = self.window_size
        
        averages = {}
        for key, values in self.history.items():
            if values:
                recent_values = list(values)[-window:]
                averages[key] = sum(recent_values) / len(recent_values)
        return averages
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all metrics."""
        stats = {}
        for key, values in self.history.items():
            if values:
                values_list = list(values)
                stats[key] = {
                    'mean': np.mean(values_list),
                    'std': np.std(values_list),
                    'min': np.min(values_list),
                    'max': np.max(values_list),
                    'count': len(values_list)
                }
        return stats
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.counts.clear()
        self.history.clear()
    
    def log(self, step: Optional[int] = None, prefix: str = "", logger: Optional[logging.Logger] = None):
        """Log current averages."""
        averages = self.get_averages()
        if averages:
            log_metrics(averages, step=step, prefix=prefix, logger=logger)


class ExperimentTracker:
    """
    Comprehensive experiment tracking with W&B integration.
    
    Features:
    - Automatic experiment naming and tagging
    - Model architecture logging
    - Training progress visualization
    - Hyperparameter sweeps support
    - Artifact management
    """
    
    def __init__(self, 
                 project_name: str = "multimodal-coconut",
                 experiment_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment tracker.
        
        Args:
            project_name: W&B project name
            experiment_name: Experiment name (auto-generated if None)
            tags: List of tags for the experiment
            config: Configuration dictionary
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.tags = tags or []
        self.config = config or {}
        
        # Initialize W&B run
        if is_main_process():
            try:
                self.run = wandb.init(
                    project=project_name,
                    name=self.experiment_name,
                    tags=self.tags,
                    config=self.config,
                    reinit=True
                )
                self.logger = get_logger()
                self.logger.info(f"Initialized experiment: {self.experiment_name}")
            except Exception as e:
                self.run = None
                self.logger = get_logger()
                self.logger.warning(f"Failed to initialize W&B: {e}")
        else:
            self.run = None
            self.logger = get_logger()
    
    def _generate_experiment_name(self) -> str:
        """Generate automatic experiment name."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"multimodal_coconut_{timestamp}"
    
    def log_model_architecture(self, model: torch.nn.Module):
        """Log model architecture and parameters."""
        if not is_main_process() or self.run is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log parameter counts
        self.run.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/frozen_parameters": total_params - trainable_params
        })
        
        # Log model summary
        model_summary = str(model)
        self.run.log({"model/architecture": wandb.Html(f"<pre>{model_summary}</pre>")})
        
        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    def log_training_metrics(self, 
                           metrics: Dict[str, Any], 
                           step: Optional[int] = None,
                           epoch: Optional[int] = None,
                           stage: Optional[int] = None):
        """Log training metrics with context."""
        if not is_main_process() or self.run is None:
            return
        
        # Add context to metrics
        log_dict = dict(metrics)
        if step is not None:
            log_dict["step"] = step
        if epoch is not None:
            log_dict["epoch"] = epoch
        if stage is not None:
            log_dict["stage"] = stage
        
        self.run.log(log_dict, step=step)
    
    def log_validation_results(self, 
                             results: Dict[str, Any],
                             step: Optional[int] = None,
                             save_artifacts: bool = True):
        """Log validation results and optionally save artifacts."""
        if not is_main_process() or self.run is None:
            return
        
        # Log metrics
        val_metrics = {f"val/{k}": v for k, v in results.items() if isinstance(v, (int, float))}
        self.run.log(val_metrics, step=step)
        
        # Save artifacts if requested
        if save_artifacts and "predictions" in results:
            self._save_predictions_artifact(results["predictions"], step)
    
    def log_stage_transition(self, 
                           from_stage: int, 
                           to_stage: int,
                           epoch: int,
                           metrics: Optional[Dict[str, Any]] = None):
        """Log curriculum stage transitions."""
        if not is_main_process() or self.run is None:
            return
        
        log_dict = {
            "curriculum/from_stage": from_stage,
            "curriculum/to_stage": to_stage,
            "curriculum/transition_epoch": epoch
        }
        
        if metrics:
            log_dict.update({f"curriculum/{k}": v for k, v in metrics.items()})
        
        self.run.log(log_dict)
        self.logger.info(f"Stage transition: {from_stage} → {to_stage} at epoch {epoch}")
    
    def log_memory_usage(self, step: Optional[int] = None):
        """Log GPU memory usage."""
        if not is_main_process() or self.run is None or not torch.cuda.is_available():
            return
        
        memory_stats = {}
        for i in range(torch.cuda.device_count()):
            memory_stats[f"memory/gpu_{i}_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
            memory_stats[f"memory/gpu_{i}_reserved_gb"] = torch.cuda.memory_reserved(i) / 1e9
            memory_stats[f"memory/gpu_{i}_max_allocated_gb"] = torch.cuda.max_memory_allocated(i) / 1e9
        
        self.run.log(memory_stats, step=step)
    
    def _save_predictions_artifact(self, predictions: List[Dict], step: Optional[int] = None):
        """Save predictions as W&B artifact."""
        try:
            # Create artifact
            artifact_name = f"predictions_step_{step}" if step else "predictions"
            artifact = wandb.Artifact(artifact_name, type="predictions")
            
            # Save predictions to temporary file
            temp_file = f"/tmp/{artifact_name}.json"
            with open(temp_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            artifact.add_file(temp_file)
            self.run.log_artifact(artifact)
            
            # Clean up
            os.remove(temp_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to save predictions artifact: {e}")
    
    def finish(self):
        """Finish the experiment."""
        if is_main_process() and self.run is not None:
            self.run.finish()
            self.logger.info("Experiment finished")


class MultimodalDebugger:
    """
    Debugging utilities for multimodal training issues.
    
    Features:
    - Batch inspection and visualization
    - Gradient flow analysis
    - Memory usage profiling
    - Model output analysis
    - Data pipeline debugging
    """
    
    def __init__(self, 
                 save_dir: str = "debug_outputs",
                 max_samples: int = 5):
        """
        Initialize multimodal debugger.
        
        Args:
            save_dir: Directory to save debug outputs
            max_samples: Maximum number of samples to debug per batch
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.logger = get_logger()
        
        # Debug counters
        self.batch_count = 0
        self.debug_count = 0
    
    def debug_batch(self, 
                   batch: Dict[str, Any],
                   tokenizer,
                   step: Optional[int] = None,
                   save_images: bool = True):
        """
        Debug a training batch.
        
        Args:
            batch: Training batch dictionary
            tokenizer: Tokenizer for text decoding
            step: Training step number
            save_images: Whether to save image visualizations
        """
        if not is_main_process():
            return
        
        self.batch_count += 1
        debug_info = {
            "batch_id": self.batch_count,
            "step": step,
            "batch_size": len(batch.get("input_ids", [])),
            "shapes": {}
        }
        
        # Analyze tensor shapes
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                debug_info["shapes"][key] = list(value.shape)
        
        # Analyze text sequences
        if "input_ids" in batch:
            self._debug_text_sequences(batch["input_ids"], tokenizer, debug_info)
        
        # Analyze images
        if "pixel_values" in batch and save_images:
            self._debug_images(batch["pixel_values"], debug_info)
        
        # Save debug info
        debug_file = self.save_dir / f"batch_debug_{self.batch_count}.json"
        with open(debug_file, 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        
        self.logger.debug(f"Saved batch debug info to {debug_file}")
    
    def _debug_text_sequences(self, 
                            input_ids: torch.Tensor, 
                            tokenizer,
                            debug_info: Dict[str, Any]):
        """Debug text sequences in the batch."""
        sequences_info = []
        
        for i, seq in enumerate(input_ids[:self.max_samples]):
            # Decode sequence
            text = tokenizer.decode(seq, skip_special_tokens=False)
            
            # Find special tokens
            special_tokens = {}
            if hasattr(tokenizer, 'latent_token_id'):
                latent_positions = (seq == tokenizer.latent_token_id).nonzero().flatten().tolist()
                special_tokens['latent_positions'] = latent_positions
            
            sequences_info.append({
                "sample_id": i,
                "sequence_length": len(seq),
                "text": text[:500] + "..." if len(text) > 500 else text,  # Truncate long sequences
                "special_tokens": special_tokens
            })
        
        debug_info["text_sequences"] = sequences_info
    
    def _debug_images(self, 
                     pixel_values: torch.Tensor,
                     debug_info: Dict[str, Any]):
        """Debug images in the batch."""
        if pixel_values.dim() == 4:  # [batch, channels, height, width]
            batch_size = pixel_values.shape[0]
        elif pixel_values.dim() == 5:  # [batch, num_patches, channels, height, width]
            batch_size = pixel_values.shape[0]
        else:
            self.logger.warning(f"Unexpected pixel_values shape: {pixel_values.shape}")
            return
        
        images_info = []
        
        for i in range(min(batch_size, self.max_samples)):
            try:
                # Extract image tensor
                if pixel_values.dim() == 5:
                    # Take first patch for visualization
                    img_tensor = pixel_values[i, 0]  # [channels, height, width]
                else:
                    img_tensor = pixel_values[i]  # [channels, height, width]
                
                # Convert to numpy and normalize for visualization
                img_np = img_tensor.cpu().numpy()
                if img_np.shape[0] == 3:  # RGB
                    img_np = np.transpose(img_np, (1, 2, 0))
                    # Normalize to [0, 1] range
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                # Save image
                img_file = self.save_dir / f"batch_{self.batch_count}_sample_{i}.png"
                plt.figure(figsize=(8, 8))
                plt.imshow(img_np)
                plt.title(f"Batch {self.batch_count}, Sample {i}")
                plt.axis('off')
                plt.savefig(img_file, bbox_inches='tight', dpi=150)
                plt.close()
                
                images_info.append({
                    "sample_id": i,
                    "image_shape": list(img_tensor.shape),
                    "image_file": str(img_file),
                    "pixel_range": [float(img_tensor.min()), float(img_tensor.max())],
                    "pixel_mean": float(img_tensor.mean()),
                    "pixel_std": float(img_tensor.std())
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to debug image {i}: {e}")
        
        debug_info["images"] = images_info
    
    def debug_model_outputs(self, 
                          outputs,
                          step: Optional[int] = None):
        """Debug model outputs."""
        if not is_main_process():
            return
        
        debug_info = {
            "step": step,
            "output_keys": list(outputs.keys()) if hasattr(outputs, 'keys') else [],
        }
        
        # Analyze loss
        if hasattr(outputs, 'loss'):
            debug_info["loss"] = {
                "value": float(outputs.loss),
                "requires_grad": outputs.loss.requires_grad,
                "grad_fn": str(outputs.loss.grad_fn) if outputs.loss.grad_fn else None
            }
        
        # Analyze logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            debug_info["logits"] = {
                "shape": list(logits.shape),
                "dtype": str(logits.dtype),
                "device": str(logits.device),
                "min": float(logits.min()),
                "max": float(logits.max()),
                "mean": float(logits.mean()),
                "std": float(logits.std()),
                "has_nan": bool(torch.isnan(logits).any()),
                "has_inf": bool(torch.isinf(logits).any())
            }
        
        # Save debug info
        debug_file = self.save_dir / f"model_outputs_step_{step}.json"
        with open(debug_file, 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        
        # Log warnings for problematic outputs
        if debug_info.get("logits", {}).get("has_nan", False):
            self.logger.warning(f"NaN detected in logits at step {step}")
        if debug_info.get("logits", {}).get("has_inf", False):
            self.logger.warning(f"Inf detected in logits at step {step}")
    
    def debug_gradients(self, 
                       model: torch.nn.Module,
                       step: Optional[int] = None):
        """Debug gradient flow in the model."""
        if not is_main_process():
            return
        
        gradient_info = {
            "step": step,
            "layers": []
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                param_count += 1
                
                gradient_info["layers"].append({
                    "name": name,
                    "shape": list(param.shape),
                    "grad_norm": grad_norm,
                    "param_norm": param.data.norm(2).item(),
                    "has_nan": bool(torch.isnan(param.grad).any()),
                    "has_inf": bool(torch.isinf(param.grad).any())
                })
        
        total_norm = total_norm ** 0.5
        gradient_info["total_grad_norm"] = total_norm
        gradient_info["param_count"] = param_count
        
        # Save gradient info
        debug_file = self.save_dir / f"gradients_step_{step}.json"
        with open(debug_file, 'w') as f:
            json.dump(gradient_info, f, indent=2, default=str)
        
        # Log warnings
        if total_norm > 100.0:
            self.logger.warning(f"Large gradient norm detected: {total_norm:.2f} at step {step}")
        
        nan_layers = [layer["name"] for layer in gradient_info["layers"] if layer["has_nan"]]
        if nan_layers:
            self.logger.warning(f"NaN gradients in layers: {nan_layers} at step {step}")
    
    def profile_memory_usage(self, step: Optional[int] = None):
        """Profile memory usage."""
        if not torch.cuda.is_available():
            return
        
        memory_info = {
            "step": step,
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                "device_id": i,
                "allocated_gb": torch.cuda.memory_allocated(i) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(i) / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated(i) / 1e9,
                "max_reserved_gb": torch.cuda.max_memory_reserved(i) / 1e9
            }
            memory_info["devices"].append(device_info)
        
        # Save memory info
        debug_file = self.save_dir / f"memory_step_{step}.json"
        with open(debug_file, 'w') as f:
            json.dump(memory_info, f, indent=2)
        
        # Log high memory usage
        for device_info in memory_info["devices"]:
            if device_info["allocated_gb"] > 10.0:  # More than 10GB
                self.logger.warning(f"High memory usage on GPU {device_info['device_id']}: "
                                  f"{device_info['allocated_gb']:.2f}GB allocated")


def create_experiment_tracker(config: Dict[str, Any]) -> ExperimentTracker:
    """
    Factory function to create an ExperimentTracker.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ExperimentTracker instance
    """
    return ExperimentTracker(
        project_name=config.get('wandb_project', 'multimodal-coconut'),
        experiment_name=config.get('experiment_name'),
        tags=config.get('tags', []),
        config=config
    )


def create_multimodal_debugger(config: Dict[str, Any]) -> MultimodalDebugger:
    """
    Factory function to create a MultimodalDebugger.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MultimodalDebugger instance
    """
    return MultimodalDebugger(
        save_dir=config.get('debug_save_dir', 'debug_outputs'),
        max_samples=config.get('debug_max_samples', 5)
    )