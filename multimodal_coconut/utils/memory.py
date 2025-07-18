"""
Memory optimization utilities for Multimodal CoCoNuT

Implements gradient checkpointing, automatic batch size reduction on OOM errors,
and KV cache optimization for multimodal sequences.
"""

import gc
import torch
import torch.nn as nn
from typing import Optional, Callable, Any, Dict, List, Tuple
from functools import wraps
import warnings

from .logging import get_logger


logger = get_logger(__name__)


class MemoryOptimizer:
    """
    Comprehensive memory optimization for multimodal CoCoNuT training.
    
    Features:
    - Gradient checkpointing for memory efficiency
    - Automatic batch size reduction on OOM errors
    - KV cache optimization for multimodal sequences
    - Memory monitoring and cleanup
    """
    
    def __init__(self, 
                 enable_gradient_checkpointing: bool = True,
                 enable_auto_batch_reduction: bool = True,
                 min_batch_size: int = 1,
                 memory_cleanup_frequency: int = 100):
        """
        Initialize memory optimizer.
        
        Args:
            enable_gradient_checkpointing: Enable gradient checkpointing
            enable_auto_batch_reduction: Enable automatic batch size reduction
            min_batch_size: Minimum allowed batch size
            memory_cleanup_frequency: Steps between memory cleanup
        """
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_auto_batch_reduction = enable_auto_batch_reduction
        self.min_batch_size = min_batch_size
        self.memory_cleanup_frequency = memory_cleanup_frequency
        
        # State tracking
        self.step_count = 0
        self.oom_count = 0
        self.current_batch_size = None
        self.original_batch_size = None
        
        # Memory statistics
        self.peak_memory_usage = 0
        self.memory_history = []
    
    def setup_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Setup gradient checkpointing for the model.
        
        Args:
            model: Model to apply gradient checkpointing to
            
        Returns:
            Model with gradient checkpointing enabled
        """
        if not self.enable_gradient_checkpointing:
            return model
        
        try:
            # Enable gradient checkpointing for transformer models
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing via model method")
            elif hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
                logger.info("Enabled input gradients for gradient checkpointing")
            
            # Apply gradient checkpointing to specific layers
            self._apply_gradient_checkpointing_to_layers(model)
            
            logger.info("Gradient checkpointing setup completed")
            
        except Exception as e:
            logger.warning(f"Could not setup gradient checkpointing: {e}")
        
        return model
    
    def _apply_gradient_checkpointing_to_layers(self, model: nn.Module):
        """Apply gradient checkpointing to specific layer types."""
        checkpointed_layers = 0
        
        for name, module in model.named_modules():
            # Apply to transformer decoder layers
            if any(layer_type in name.lower() for layer_type in 
                   ['decoder_layer', 'transformer_block', 'attention_block']):
                if hasattr(module, 'forward'):
                    module.forward = torch.utils.checkpoint.checkpoint(
                        module.forward, use_reentrant=False
                    )
                    checkpointed_layers += 1
        
        if checkpointed_layers > 0:
            logger.info(f"Applied gradient checkpointing to {checkpointed_layers} layers")
    
    def optimize_batch_for_memory(self, 
                                 batch: Dict[str, Any], 
                                 target_batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize batch for memory usage.
        
        Args:
            batch: Input batch
            target_batch_size: Target batch size (optional)
            
        Returns:
            Optimized batch
        """
        if target_batch_size is None:
            return batch
        
        current_batch_size = self._get_batch_size(batch)
        
        if current_batch_size <= target_batch_size:
            return batch
        
        # Reduce batch size
        optimized_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                optimized_batch[key] = value[:target_batch_size]
            elif isinstance(value, list) and len(value) > target_batch_size:
                optimized_batch[key] = value[:target_batch_size]
            else:
                optimized_batch[key] = value
        
        logger.info(f"Reduced batch size from {current_batch_size} to {target_batch_size}")
        return optimized_batch
    
    def handle_oom_error(self, 
                        batch: Dict[str, Any], 
                        forward_fn: Callable,
                        reduction_factor: float = 0.5) -> Tuple[Any, Dict[str, Any]]:
        """
        Handle OOM error by reducing batch size and retrying.
        
        Args:
            batch: Input batch that caused OOM
            forward_fn: Forward function to retry
            reduction_factor: Factor to reduce batch size by
            
        Returns:
            Tuple of (forward_result, optimized_batch)
        """
        if not self.enable_auto_batch_reduction:
            raise RuntimeError("OOM error occurred and auto batch reduction is disabled")
        
        current_batch_size = self._get_batch_size(batch)
        
        if self.original_batch_size is None:
            self.original_batch_size = current_batch_size
        
        # Calculate new batch size
        new_batch_size = max(
            self.min_batch_size,
            int(current_batch_size * reduction_factor)
        )
        
        if new_batch_size >= current_batch_size:
            raise RuntimeError(f"Cannot reduce batch size further (current: {current_batch_size}, min: {self.min_batch_size})")
        
        self.oom_count += 1
        logger.warning(f"OOM error #{self.oom_count}: Reducing batch size from {current_batch_size} to {new_batch_size}")
        
        # Clean up memory
        self.cleanup_memory()
        
        # Optimize batch
        optimized_batch = self.optimize_batch_for_memory(batch, new_batch_size)
        
        # Retry forward pass
        try:
            result = forward_fn(optimized_batch)
            self.current_batch_size = new_batch_size
            logger.info(f"Successfully recovered from OOM with batch size {new_batch_size}")
            return result, optimized_batch
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Recursive retry with further reduction
                return self.handle_oom_error(optimized_batch, forward_fn, reduction_factor)
            else:
                raise e
    
    def optimize_kv_cache(self, 
                         kv_cache: Optional[Dict[str, torch.Tensor]],
                         sequence_length: int,
                         max_cache_length: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Optimize KV cache for multimodal sequences.
        
        Args:
            kv_cache: Key-Value cache dictionary
            sequence_length: Current sequence length
            max_cache_length: Maximum cache length to maintain
            
        Returns:
            Optimized KV cache
        """
        if kv_cache is None:
            return None
        
        if max_cache_length is None:
            max_cache_length = sequence_length * 2  # Default heuristic
        
        optimized_cache = {}
        
        for key, cache_tensor in kv_cache.items():
            if isinstance(cache_tensor, torch.Tensor):
                # Trim cache if it's too long
                if cache_tensor.size(-2) > max_cache_length:
                    # Keep the most recent entries
                    start_idx = cache_tensor.size(-2) - max_cache_length
                    optimized_cache[key] = cache_tensor[..., start_idx:, :]
                    logger.debug(f"Trimmed KV cache for {key} from {cache_tensor.size(-2)} to {max_cache_length}")
                else:
                    optimized_cache[key] = cache_tensor
            else:
                optimized_cache[key] = cache_tensor
        
        return optimized_cache
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        # Clear Python garbage
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("Memory cleanup completed")
    
    def monitor_memory_usage(self):
        """Monitor and log memory usage."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            
            self.peak_memory_usage = max(self.peak_memory_usage, current_memory)
            self.memory_history.append(current_memory)
            
            # Keep only recent history
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]
            
            logger.debug(f"Memory usage: {current_memory / 1e9:.2f}GB / {max_memory / 1e9:.2f}GB")
    
    def step(self):
        """Called at each training step for memory management."""
        self.step_count += 1
        
        # Periodic memory cleanup
        if self.step_count % self.memory_cleanup_frequency == 0:
            self.cleanup_memory()
        
        # Monitor memory usage
        self.monitor_memory_usage()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'oom_count': self.oom_count,
            'current_batch_size': self.current_batch_size,
            'original_batch_size': self.original_batch_size,
            'peak_memory_usage': self.peak_memory_usage,
            'step_count': self.step_count
        }
        
        if torch.cuda.is_available():
            stats.update({
                'current_memory_allocated': torch.cuda.memory_allocated(),
                'max_memory_allocated': torch.cuda.max_memory_allocated(),
                'memory_reserved': torch.cuda.memory_reserved(),
                'max_memory_reserved': torch.cuda.max_memory_reserved()
            })
        
        return stats
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """Extract batch size from batch dictionary."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value.size(0)
            elif isinstance(value, list):
                return len(value)
        return 1


def memory_efficient_forward(memory_optimizer: MemoryOptimizer):
    """
    Decorator for memory-efficient forward passes.
    
    Args:
        memory_optimizer: MemoryOptimizer instance
        
    Returns:
        Decorated function
    """
    def decorator(forward_fn):
        @wraps(forward_fn)
        def wrapper(*args, **kwargs):
            try:
                # Monitor memory before forward pass
                memory_optimizer.monitor_memory_usage()
                
                # Execute forward pass
                result = forward_fn(*args, **kwargs)
                
                # Update step counter
                memory_optimizer.step()
                
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error
                    if len(args) > 0 and isinstance(args[0], dict):
                        batch = args[0]
                        result, optimized_batch = memory_optimizer.handle_oom_error(
                            batch, lambda b: forward_fn(b, *args[1:], **kwargs)
                        )
                        return result
                    else:
                        raise e
                else:
                    raise e
        
        return wrapper
    return decorator


def optimize_multimodal_kv_cache(kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
                                max_length: int = 2048,
                                compression_ratio: float = 0.5) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Optimize KV cache for multimodal sequences with compression.
    
    Args:
        kv_cache: List of (key, value) tensor tuples for each layer
        max_length: Maximum sequence length to maintain
        compression_ratio: Ratio of cache to compress when exceeding max_length
        
    Returns:
        Optimized KV cache
    """
    if kv_cache is None:
        return None
    
    optimized_cache = []
    
    for layer_idx, (key_cache, value_cache) in enumerate(kv_cache):
        if key_cache is None or value_cache is None:
            optimized_cache.append((key_cache, value_cache))
            continue
        
        seq_len = key_cache.size(-2)
        
        if seq_len > max_length:
            # Calculate how much to keep
            keep_length = int(max_length * (1 - compression_ratio))
            compress_length = max_length - keep_length
            
            # Keep recent tokens and compress older ones
            if compress_length > 0:
                # Simple compression: average pooling over time dimension
                old_keys = key_cache[..., :seq_len-keep_length, :]
                old_values = value_cache[..., :seq_len-keep_length, :]
                
                # Reshape for pooling
                batch_size, num_heads, old_seq_len, head_dim = old_keys.shape
                pool_size = old_seq_len // compress_length
                
                if pool_size > 1:
                    # Average pool to compress
                    compressed_keys = old_keys[..., :pool_size*compress_length, :].view(
                        batch_size, num_heads, compress_length, pool_size, head_dim
                    ).mean(dim=3)
                    
                    compressed_values = old_values[..., :pool_size*compress_length, :].view(
                        batch_size, num_heads, compress_length, pool_size, head_dim
                    ).mean(dim=3)
                else:
                    compressed_keys = old_keys[..., :compress_length, :]
                    compressed_values = old_values[..., :compress_length, :]
                
                # Keep recent tokens
                recent_keys = key_cache[..., -keep_length:, :]
                recent_values = value_cache[..., -keep_length:, :]
                
                # Concatenate compressed and recent
                optimized_key = torch.cat([compressed_keys, recent_keys], dim=-2)
                optimized_value = torch.cat([compressed_values, recent_values], dim=-2)
            else:
                # Just truncate to max_length
                optimized_key = key_cache[..., -max_length:, :]
                optimized_value = value_cache[..., -max_length:, :]
            
            optimized_cache.append((optimized_key, optimized_value))
            
            logger.debug(f"Layer {layer_idx}: Compressed KV cache from {seq_len} to {optimized_key.size(-2)}")
        else:
            optimized_cache.append((key_cache, value_cache))
    
    return optimized_cache


def create_memory_optimizer(**kwargs) -> MemoryOptimizer:
    """
    Factory function to create a MemoryOptimizer.
    
    Args:
        **kwargs: Arguments for MemoryOptimizer
        
    Returns:
        Configured MemoryOptimizer instance
    """
    return MemoryOptimizer(**kwargs)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage in GB
    """
    stats = {}
    
    if torch.cuda.is_available():
        stats.update({
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1e9
        })
    else:
        stats.update({
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'max_allocated_gb': 0.0,
            'max_reserved_gb': 0.0
        })
    
    return stats


def reset_memory_stats():
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()