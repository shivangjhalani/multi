"""
Distributed training utilities for Multimodal CoCoNuT

Following the original CoCoNuT's distributed training patterns with enhanced
FSDP and DDP support for multimodal training.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Any, Dict
import functools


def init_distributed_training() -> tuple[int, int, int]:
    """
    Initialize distributed training environment.
    
    Returns:
        Tuple of (local_rank, rank, world_size)
    """
    # Initialize distributed environment
    if not dist.is_initialized():
        # Use NCCL backend for GPU training
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    
    # Get distributed training parameters
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set CUDA device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def setup_distributed_environment() -> Dict[str, Any]:
    """
    Setup comprehensive distributed training environment.
    
    Returns:
        Dictionary with distributed training information
    """
    # Check if distributed training is available
    if not torch.distributed.is_available():
        return {
            'distributed': False,
            'local_rank': 0,
            'rank': 0,
            'world_size': 1,
            'backend': None
        }
    
    # Initialize if environment variables are set
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank, rank, world_size = init_distributed_training()
        backend = dist.get_backend()
        
        return {
            'distributed': True,
            'local_rank': local_rank,
            'rank': rank,
            'world_size': world_size,
            'backend': backend
        }
    else:
        return {
            'distributed': False,
            'local_rank': 0,
            'rank': 0,
            'world_size': 1,
            'backend': None
        }


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor


def gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensor from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def setup_ddp_model(model: torch.nn.Module, device_ids: Optional[list] = None) -> torch.nn.Module:
    """
    Setup model for Distributed Data Parallel training.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs
        
    Returns:
        DDP wrapped model
    """
    if not dist.is_initialized():
        return model
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    if device_ids is None:
        device_ids = [torch.cuda.current_device()]
    
    return DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
        find_unused_parameters=True
    )


def setup_fsdp_model(model: torch.nn.Module, 
                     auto_wrap_policy=None,
                     mixed_precision_policy=None,
                     sharding_strategy=None,
                     cpu_offload=None) -> torch.nn.Module:
    """
    Setup model for Fully Sharded Data Parallel training with enhanced multimodal support.
    
    Args:
        model: Model to wrap
        auto_wrap_policy: Auto wrap policy for FSDP
        mixed_precision_policy: Mixed precision policy
        sharding_strategy: Sharding strategy for FSDP
        cpu_offload: CPU offload configuration
        
    Returns:
        FSDP wrapped model
    """
    if not dist.is_initialized():
        return model
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp import MixedPrecision
    import functools
    
    # Default auto wrap policy for transformer models
    if auto_wrap_policy is None:
        try:
            # Try to import InternVL and common transformer layer types
            layer_classes = set()
            
            # InternVL specific layers
            try:
                from transformers.models.internlm2.modeling_internlm2 import InternLM2DecoderLayer
                layer_classes.add(InternLM2DecoderLayer)
            except ImportError:
                pass
            
            # Common transformer layers
            try:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                layer_classes.add(LlamaDecoderLayer)
            except ImportError:
                pass
                
            try:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                layer_classes.add(GPT2Block)
            except ImportError:
                pass
            
            if layer_classes:
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=layer_classes
                )
            else:
                # Fallback to size-based wrapping
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy,
                    min_num_params=1e6  # 1M parameters
                )
        except ImportError:
            # Final fallback
            auto_wrap_policy = None
    
    # Default sharding strategy
    if sharding_strategy is None:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    
    # Default mixed precision policy for multimodal training
    if mixed_precision_policy is None:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # Default CPU offload (disabled by default for performance)
    if cpu_offload is None:
        cpu_offload = CPUOffload(offload_params=False)
    
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,  # Ensure consistent initialization across ranks
        forward_prefetch=True,   # Prefetch parameters for better performance
        backward_prefetch=True,  # Prefetch gradients for better performance
    )


def setup_multimodal_distributed_model(model: torch.nn.Module,
                                     strategy: str = "fsdp",
                                     **kwargs) -> torch.nn.Module:
    """
    Setup model for distributed training with multimodal-specific optimizations.
    
    Args:
        model: Model to wrap
        strategy: Distribution strategy ("fsdp" or "ddp")
        **kwargs: Additional arguments for the specific strategy
        
    Returns:
        Distributed model
    """
    if not dist.is_initialized():
        return model
    
    if strategy.lower() == "fsdp":
        return setup_fsdp_model(model, **kwargs)
    elif strategy.lower() == "ddp":
        return setup_ddp_model(model, **kwargs)
    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")


def synchronize_multimodal_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize multimodal batch across processes for consistent training.
    
    This ensures that all processes have consistent batch information,
    particularly important for multimodal data with variable image sizes.
    
    Args:
        batch: Multimodal batch dictionary
        
    Returns:
        Synchronized batch
    """
    if not dist.is_initialized():
        return batch
    
    # Synchronize batch size information
    if 'input_ids' in batch:
        batch_size = batch['input_ids'].size(0)
        batch_size_tensor = torch.tensor(batch_size, device=batch['input_ids'].device)
        dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.MAX)
        
        # Store synchronized batch size for reference
        batch['_sync_batch_size'] = batch_size_tensor.item()
    
    # Synchronize sequence length information
    if 'input_ids' in batch:
        seq_len = batch['input_ids'].size(1)
        seq_len_tensor = torch.tensor(seq_len, device=batch['input_ids'].device)
        dist.all_reduce(seq_len_tensor, op=dist.ReduceOp.MAX)
        
        # Store synchronized sequence length for reference
        batch['_sync_seq_len'] = seq_len_tensor.item()
    
    # Synchronize number of image patches information
    if '_num_patches_list' in batch:
        max_patches = max(batch['_num_patches_list'])
        max_patches_tensor = torch.tensor(max_patches, device=next(iter(batch.values())).device)
        dist.all_reduce(max_patches_tensor, op=dist.ReduceOp.MAX)
        
        # Store synchronized max patches for reference
        batch['_sync_max_patches'] = max_patches_tensor.item()
    
    return batch


def get_distributed_sampler(dataset, shuffle: bool = True, drop_last: bool = False):
    """
    Get appropriate distributed sampler for the dataset.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Distributed sampler or None if not in distributed mode
    """
    if not dist.is_initialized():
        return None
    
    from torch.utils.data.distributed import DistributedSampler
    
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last
    )


def all_gather_object(obj: Any) -> list:
    """
    Gather objects from all processes.
    
    Args:
        obj: Object to gather
        
    Returns:
        List of objects from all processes
    """
    if not dist.is_initialized():
        return [obj]
    
    world_size = get_world_size()
    gathered_objects = [None] * world_size
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast object from source rank to all other ranks.
    
    Args:
        obj: Object to broadcast (only used on src rank)
        src: Source rank
        
    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj
    
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]