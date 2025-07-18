"""
Debugging utilities for multimodal training issues

This module provides specialized debugging tools for multimodal CoCoNuT training:
- Data pipeline debugging and visualization
- Model behavior analysis
- Training convergence monitoring
- Performance profiling
- Error diagnosis and recovery
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict, deque
import logging

from .logging import get_logger
from .distributed import is_main_process


class DataPipelineDebugger:
    """
    Debug data pipeline issues in multimodal training.
    
    Features:
    - Batch composition analysis
    - Image quality assessment
    - Text tokenization validation
    - Special token placement verification
    """
    
    def __init__(self, save_dir: str = "debug/data_pipeline"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Statistics tracking
        self.batch_stats = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def analyze_batch_composition(self, 
                                batch: Dict[str, Any],
                                batch_idx: int,
                                tokenizer) -> Dict[str, Any]:
        """
        Analyze the composition of a training batch.
        
        Args:
            batch: Training batch dictionary
            batch_idx: Batch index
            tokenizer: Tokenizer for text analysis
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            "batch_idx": batch_idx,
            "timestamp": time.time(),
            "batch_size": 0,
            "sequence_lengths": [],
            "image_info": {},
            "text_info": {},
            "special_tokens": {},
            "issues": []
        }
        
        # Analyze batch size
        if "input_ids" in batch:
            analysis["batch_size"] = batch["input_ids"].shape[0]
            analysis["sequence_lengths"] = batch["input_ids"].shape[1:] if len(batch["input_ids"].shape) > 1 else []
        
        # Analyze images
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
            analysis["image_info"] = {
                "shape": list(pixel_values.shape),
                "dtype": str(pixel_values.dtype),
                "device": str(pixel_values.device),
                "pixel_range": [float(pixel_values.min()), float(pixel_values.max())],
                "has_nan": bool(torch.isnan(pixel_values).any()),
                "has_inf": bool(torch.isinf(pixel_values).any())
            }
            
            # Check for image issues
            if analysis["image_info"]["has_nan"]:
                analysis["issues"].append("NaN values in pixel_values")
                self.error_counts["nan_pixels"] += 1
            
            if analysis["image_info"]["has_inf"]:
                analysis["issues"].append("Inf values in pixel_values")
                self.error_counts["inf_pixels"] += 1
        
        # Analyze text sequences
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            analysis["text_info"] = {
                "vocab_size": tokenizer.vocab_size,
                "sequence_length": input_ids.shape[1] if len(input_ids.shape) > 1 else 0,
                "padding_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
            
            # Analyze special tokens
            special_token_analysis = self._analyze_special_tokens(input_ids, tokenizer)
            analysis["special_tokens"] = special_token_analysis
            
            # Check for text issues
            if (input_ids >= tokenizer.vocab_size).any():
                analysis["issues"].append("Token IDs exceed vocabulary size")
                self.error_counts["invalid_tokens"] += 1
        
        # Save analysis
        if is_main_process():
            analysis_file = self.save_dir / f"batch_analysis_{batch_idx}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        
        # Update statistics
        self.batch_stats["batch_sizes"].append(analysis["batch_size"])
        if analysis["issues"]:
            self.batch_stats["issues_per_batch"].append(len(analysis["issues"]))
        
        return analysis
    
    def _analyze_special_tokens(self, 
                              input_ids: torch.Tensor,
                              tokenizer) -> Dict[str, Any]:
        """Analyze special token placement in sequences."""
        special_analysis = {
            "latent_tokens": {},
            "start_tokens": {},
            "end_tokens": {},
            "patterns": []
        }
        
        # Check for CoCoNuT special tokens
        if hasattr(tokenizer, 'latent_token_id'):
            latent_positions = (input_ids == tokenizer.latent_token_id).nonzero()
            special_analysis["latent_tokens"] = {
                "count": len(latent_positions),
                "positions": latent_positions.tolist() if len(latent_positions) < 100 else "too_many"
            }
        
        if hasattr(tokenizer, 'start_latent_id'):
            start_positions = (input_ids == tokenizer.start_latent_id).nonzero()
            special_analysis["start_tokens"] = {
                "count": len(start_positions),
                "positions": start_positions.tolist() if len(start_positions) < 100 else "too_many"
            }
        
        if hasattr(tokenizer, 'end_latent_id'):
            end_positions = (input_ids == tokenizer.end_latent_id).nonzero()
            special_analysis["end_tokens"] = {
                "count": len(end_positions),
                "positions": end_positions.tolist() if len(end_positions) < 100 else "too_many"
            }
        
        return special_analysis
    
    def visualize_batch_statistics(self, save_plots: bool = True) -> Dict[str, Any]:
        """Create visualizations of batch statistics."""
        if not is_main_process() or not self.batch_stats:
            return {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Data Pipeline Statistics", fontsize=16)
        
        # Batch size distribution
        if self.batch_stats["batch_sizes"]:
            axes[0, 0].hist(self.batch_stats["batch_sizes"], bins=20, alpha=0.7)
            axes[0, 0].set_title("Batch Size Distribution")
            axes[0, 0].set_xlabel("Batch Size")
            axes[0, 0].set_ylabel("Frequency")
        
        # Issues per batch
        if self.batch_stats["issues_per_batch"]:
            axes[0, 1].hist(self.batch_stats["issues_per_batch"], bins=10, alpha=0.7, color='red')
            axes[0, 1].set_title("Issues per Batch")
            axes[0, 1].set_xlabel("Number of Issues")
            axes[0, 1].set_ylabel("Frequency")
        
        # Error counts
        if self.error_counts:
            error_types = list(self.error_counts.keys())
            error_counts = list(self.error_counts.values())
            axes[1, 0].bar(error_types, error_counts)
            axes[1, 0].set_title("Error Type Counts")
            axes[1, 0].set_xlabel("Error Type")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Summary statistics
        summary_text = f"""
        Total Batches Analyzed: {len(self.batch_stats['batch_sizes'])}
        Average Batch Size: {np.mean(self.batch_stats['batch_sizes']):.2f}
        Total Errors: {sum(self.error_counts.values())}
        Error Rate: {sum(self.error_counts.values()) / max(len(self.batch_stats['batch_sizes']), 1):.3f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.save_dir / "batch_statistics.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved batch statistics plot to {plot_file}")
        
        plt.close()
        
        return {
            "total_batches": len(self.batch_stats["batch_sizes"]),
            "avg_batch_size": np.mean(self.batch_stats["batch_sizes"]) if self.batch_stats["batch_sizes"] else 0,
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts)
        }


class ModelBehaviorAnalyzer:
    """
    Analyze model behavior during training.
    
    Features:
    - Activation analysis
    - Gradient flow monitoring
    - Loss landscape visualization
    - Attention pattern analysis
    """
    
    def __init__(self, save_dir: str = "debug/model_behavior"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Tracking data
        self.activation_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.loss_history = []
    
    def analyze_activations(self, 
                          model: torch.nn.Module,
                          step: int,
                          sample_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze model activations.
        
        Args:
            model: Model to analyze
            step: Training step
            sample_layers: Specific layers to analyze (None for all)
            
        Returns:
            Activation analysis results
        """
        if not is_main_process():
            return {}
        
        activation_analysis = {
            "step": step,
            "timestamp": time.time(),
            "layers": {}
        }
        
        # Hook function to capture activations
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if sample_layers is None or name in sample_layers:
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        
        # Forward pass to capture activations (this should be called during training)
        # Note: This is a placeholder - actual activation capture happens during forward pass
        
        # Analyze captured activations
        for layer_name, activation in activations.items():
            if activation.numel() > 0:
                stats = {
                    "shape": list(activation.shape),
                    "mean": float(activation.mean()),
                    "std": float(activation.std()),
                    "min": float(activation.min()),
                    "max": float(activation.max()),
                    "zero_fraction": float((activation == 0).float().mean()),
                    "has_nan": bool(torch.isnan(activation).any()),
                    "has_inf": bool(torch.isinf(activation).any())
                }
                activation_analysis["layers"][layer_name] = stats
                self.activation_stats[layer_name].append(stats)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Save analysis
        analysis_file = self.save_dir / f"activations_step_{step}.json"
        with open(analysis_file, 'w') as f:
            json.dump(activation_analysis, f, indent=2, default=str)
        
        return activation_analysis
    
    def analyze_gradient_flow(self, 
                            model: torch.nn.Module,
                            step: int) -> Dict[str, Any]:
        """
        Analyze gradient flow through the model.
        
        Args:
            model: Model to analyze
            step: Training step
            
        Returns:
            Gradient flow analysis
        """
        if not is_main_process():
            return {}
        
        gradient_analysis = {
            "step": step,
            "timestamp": time.time(),
            "layers": {},
            "summary": {}
        }
        
        total_norm = 0.0
        param_count = 0
        zero_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()
                
                layer_stats = {
                    "param_shape": list(param.shape),
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_to_param_ratio": grad_norm / (param_norm + 1e-8),
                    "has_nan": bool(torch.isnan(param.grad).any()),
                    "has_inf": bool(torch.isinf(param.grad).any()),
                    "is_zero": grad_norm < 1e-8
                }
                
                gradient_analysis["layers"][name] = layer_stats
                self.gradient_stats[name].append(layer_stats)
                
                total_norm += grad_norm ** 2
                param_count += 1
                
                if layer_stats["is_zero"]:
                    zero_grad_count += 1
        
        # Summary statistics
        total_norm = total_norm ** 0.5
        gradient_analysis["summary"] = {
            "total_grad_norm": total_norm,
            "param_count": param_count,
            "zero_grad_count": zero_grad_count,
            "zero_grad_fraction": zero_grad_count / max(param_count, 1)
        }
        
        # Save analysis
        analysis_file = self.save_dir / f"gradients_step_{step}.json"
        with open(analysis_file, 'w') as f:
            json.dump(gradient_analysis, f, indent=2, default=str)
        
        # Log warnings
        if total_norm > 100.0:
            self.logger.warning(f"Large gradient norm: {total_norm:.2f} at step {step}")
        
        if gradient_analysis["summary"]["zero_grad_fraction"] > 0.5:
            self.logger.warning(f"High zero gradient fraction: "
                              f"{gradient_analysis['summary']['zero_grad_fraction']:.2f} at step {step}")
        
        return gradient_analysis
    
    def track_loss_landscape(self, 
                           loss: float,
                           step: int,
                           additional_metrics: Optional[Dict[str, float]] = None):
        """
        Track loss landscape over training.
        
        Args:
            loss: Current loss value
            step: Training step
            additional_metrics: Additional metrics to track
        """
        loss_entry = {
            "step": step,
            "loss": loss,
            "timestamp": time.time()
        }
        
        if additional_metrics:
            loss_entry.update(additional_metrics)
        
        self.loss_history.append(loss_entry)
        
        # Periodically save and visualize
        if step % 1000 == 0 and is_main_process():
            self._visualize_loss_landscape()
    
    def _visualize_loss_landscape(self):
        """Create loss landscape visualization."""
        if not self.loss_history:
            return
        
        steps = [entry["step"] for entry in self.loss_history]
        losses = [entry["loss"] for entry in self.loss_history]
        
        plt.figure(figsize=(12, 6))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, alpha=0.7)
        plt.title("Loss Landscape")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        # Loss distribution
        plt.subplot(1, 2, 2)
        plt.hist(losses, bins=50, alpha=0.7)
        plt.title("Loss Distribution")
        plt.xlabel("Loss Value")
        plt.ylabel("Frequency")
        plt.yscale('log')
        
        plt.tight_layout()
        
        plot_file = self.save_dir / "loss_landscape.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Updated loss landscape visualization: {plot_file}")


class TrainingMonitor:
    """
    Monitor training progress and detect issues.
    
    Features:
    - Convergence monitoring
    - Performance regression detection
    - Resource utilization tracking
    - Early stopping recommendations
    """
    
    def __init__(self, 
                 save_dir: str = "debug/training_monitor",
                 patience: int = 10,
                 min_delta: float = 1e-4):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Monitoring parameters
        self.patience = patience
        self.min_delta = min_delta
        
        # Tracking data
        self.metrics_history = defaultdict(list)
        self.alerts = []
        self.best_metrics = {}
        self.no_improvement_count = defaultdict(int)
    
    def update_metrics(self, 
                      metrics: Dict[str, float],
                      step: int,
                      epoch: Optional[int] = None):
        """
        Update training metrics and check for issues.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch (optional)
        """
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            entry = {
                "step": step,
                "epoch": epoch,
                "value": value,
                "timestamp": timestamp
            }
            self.metrics_history[metric_name].append(entry)
            
            # Check for improvements
            self._check_improvement(metric_name, value, step)
            
            # Check for anomalies
            self._check_anomalies(metric_name, value, step)
        
        # Periodic analysis
        if step % 500 == 0 and is_main_process():
            self._analyze_trends()
    
    def _check_improvement(self, metric_name: str, value: float, step: int):
        """Check if metric has improved."""
        is_loss_metric = 'loss' in metric_name.lower()
        
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = value
            self.no_improvement_count[metric_name] = 0
            return
        
        improved = False
        if is_loss_metric:
            # Lower is better for loss metrics
            if value < self.best_metrics[metric_name] - self.min_delta:
                improved = True
                self.best_metrics[metric_name] = value
        else:
            # Higher is better for other metrics
            if value > self.best_metrics[metric_name] + self.min_delta:
                improved = True
                self.best_metrics[metric_name] = value
        
        if improved:
            self.no_improvement_count[metric_name] = 0
        else:
            self.no_improvement_count[metric_name] += 1
            
            # Check for early stopping
            if self.no_improvement_count[metric_name] >= self.patience:
                alert = {
                    "type": "no_improvement",
                    "metric": metric_name,
                    "step": step,
                    "count": self.no_improvement_count[metric_name],
                    "message": f"No improvement in {metric_name} for {self.no_improvement_count[metric_name]} steps"
                }
                self.alerts.append(alert)
                self.logger.warning(alert["message"])
    
    def _check_anomalies(self, metric_name: str, value: float, step: int):
        """Check for anomalous metric values."""
        history = self.metrics_history[metric_name]
        
        if len(history) < 10:  # Need some history
            return
        
        recent_values = [entry["value"] for entry in history[-10:]]
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        # Check for outliers (more than 3 standard deviations)
        if abs(value - mean_val) > 3 * std_val and std_val > 0:
            alert = {
                "type": "anomaly",
                "metric": metric_name,
                "step": step,
                "value": value,
                "expected_range": [mean_val - 2*std_val, mean_val + 2*std_val],
                "message": f"Anomalous {metric_name} value: {value:.4f} (expected: {mean_val:.4f} ± {2*std_val:.4f})"
            }
            self.alerts.append(alert)
            self.logger.warning(alert["message"])
        
        # Check for NaN or Inf
        if np.isnan(value) or np.isinf(value):
            alert = {
                "type": "invalid_value",
                "metric": metric_name,
                "step": step,
                "value": value,
                "message": f"Invalid {metric_name} value: {value}"
            }
            self.alerts.append(alert)
            self.logger.error(alert["message"])
    
    def _analyze_trends(self):
        """Analyze metric trends."""
        if not is_main_process():
            return
        
        trend_analysis = {
            "timestamp": time.time(),
            "metrics": {}
        }
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 20:  # Need sufficient history
                continue
            
            values = [entry["value"] for entry in history[-20:]]
            steps = [entry["step"] for entry in history[-20:]]
            
            # Calculate trend
            if len(values) > 1:
                slope = np.polyfit(steps, values, 1)[0]
                trend_analysis["metrics"][metric_name] = {
                    "slope": slope,
                    "recent_mean": np.mean(values),
                    "recent_std": np.std(values),
                    "trend": "improving" if slope < 0 and 'loss' in metric_name.lower() else 
                            "improving" if slope > 0 and 'loss' not in metric_name.lower() else
                            "degrading" if slope > 0 and 'loss' in metric_name.lower() else
                            "degrading" if slope < 0 and 'loss' not in metric_name.lower() else
                            "stable"
                }
        
        # Save trend analysis
        analysis_file = self.save_dir / f"trend_analysis_{int(time.time())}.json"
        with open(analysis_file, 'w') as f:
            json.dump(trend_analysis, f, indent=2, default=str)
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get training recommendations based on monitoring."""
        recommendations = []
        
        # Check for metrics that haven't improved
        for metric_name, count in self.no_improvement_count.items():
            if count >= self.patience // 2:
                recommendations.append({
                    "type": "early_stopping_warning",
                    "metric": metric_name,
                    "message": f"Consider early stopping - {metric_name} hasn't improved for {count} steps",
                    "severity": "medium" if count < self.patience else "high"
                })
        
        # Check recent alerts
        recent_alerts = [alert for alert in self.alerts if time.time() - alert.get("timestamp", 0) < 3600]  # Last hour
        if len(recent_alerts) > 10:
            recommendations.append({
                "type": "high_alert_frequency",
                "message": f"High number of alerts in the last hour: {len(recent_alerts)}",
                "severity": "high"
            })
        
        return recommendations


def create_comprehensive_debugger(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comprehensive debugging suite.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of debugging tools
    """
    debug_dir = config.get('debug_save_dir', 'debug_outputs')
    
    return {
        'data_pipeline': DataPipelineDebugger(save_dir=f"{debug_dir}/data_pipeline"),
        'model_behavior': ModelBehaviorAnalyzer(save_dir=f"{debug_dir}/model_behavior"),
        'training_monitor': TrainingMonitor(
            save_dir=f"{debug_dir}/training_monitor",
            patience=config.get('early_stopping_patience', 10),
            min_delta=config.get('early_stopping_min_delta', 1e-4)
        )
    }