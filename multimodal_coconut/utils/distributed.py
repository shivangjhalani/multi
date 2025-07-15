"""
Distributed training utilities for Multimodal CoCoNuT

Following the original CoCoNuT's distributed training patterns.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Any


def init_distributed_training() -> tuple[int, int, int]:
    """
    Initialize distributed training environment.
    
    Returns:
        Tuple of (local_rank, rank, world_size)
    """
    # Initialize distributed environment
    dist.init_process_group("nccl")
    
    # Get distributed training parameters
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


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


def setup_fsdp_model(model: torch.nn.Module, auto_wrap_policy=None) -> torch.nn.Module:
    """
    Setup model for Fully Sharded Data Parallel training.
    
    Args:
        model: Model to wrap
        auto_wrap_policy: Auto wrap policy for FSDP
        
    Returns:
        FSDP wrapped model
    """
    if not dist.is_initialized():
        return model
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools
    
    # Default auto wrap policy for transformer models
    if auto_wrap_policy is None:
        try:
            # Try to import common transformer layer types
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer, GPT2Block}
            )
        except ImportError:
            # Fallback to size-based wrapping
            auto_wrap_policy = None
    
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=None,  # Can be configured based on requirements
        device_id=torch.cuda.current_device()
    )