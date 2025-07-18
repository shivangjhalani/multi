"""
Utility functions for Multimodal CoCoNuT
"""

from .distributed import (
    init_distributed_training,
    setup_distributed_environment,
    setup_multimodal_distributed_model,
    synchronize_multimodal_batch,
    get_distributed_sampler,
    all_gather_object,
    broadcast_object,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    reduce_tensor,
    gather_tensor
)

from .logging import (
    setup_logging,
    get_logger,
    log_config,
    log_metrics
)

from .misc import (
    set_seed,
    count_parameters,
    get_device,
    create_optimizer,
    create_scheduler
)

from .checkpoint import (
    CheckpointManager,
    create_checkpoint_manager,
    auto_resume_from_checkpoint
)

from .memory import (
    MemoryOptimizer,
    memory_efficient_forward,
    optimize_multimodal_kv_cache,
    create_memory_optimizer,
    get_memory_usage,
    reset_memory_stats
)

__all__ = [
    # Distributed training
    'init_distributed_training',
    'setup_distributed_environment',
    'setup_multimodal_distributed_model',
    'synchronize_multimodal_batch',
    'get_distributed_sampler',
    'all_gather_object',
    'broadcast_object',
    'cleanup_distributed', 
    'is_main_process',
    'get_rank',
    'get_world_size',
    'barrier',
    'reduce_tensor',
    'gather_tensor',
    
    # Logging
    'setup_logging',
    'get_logger',
    'log_config',
    'log_metrics',
    
    # Miscellaneous utilities
    'set_seed',
    'count_parameters',
    'get_device',
    'create_optimizer',
    'create_scheduler',
    
    # Checkpoint management
    'CheckpointManager',
    'create_checkpoint_manager',
    'auto_resume_from_checkpoint',
    
    # Memory optimization
    'MemoryOptimizer',
    'memory_efficient_forward',
    'optimize_multimodal_kv_cache',
    'create_memory_optimizer',
    'get_memory_usage',
    'reset_memory_stats'
]