# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Dataset utilities for monitoring and debugging multimodal data processing

import time
import psutil
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ProcessingMonitor:
    """Monitor dataset processing for hangs and resource usage"""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.last_update = None
        self.processed_count = 0
        self.total_count = 0
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, total_count: int):
        """Start monitoring dataset processing"""
        self.start_time = time.time()
        self.last_update = self.start_time
        self.processed_count = 0
        self.total_count = total_count
        self.is_monitoring = True
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started monitoring {total_count} samples with {self.timeout_seconds}s timeout")
    
    def update_progress(self, count: int):
        """Update processing progress"""
        self.processed_count = count
        self.last_update = time.time()
        
        if count % 100 == 0:
            elapsed = time.time() - self.start_time
            rate = count / elapsed if elapsed > 0 else 0
            logger.info(f"Processed {count}/{self.total_count} samples ({rate:.1f} samples/sec)")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Processing completed in {elapsed:.1f}s")
    
    def _monitor_loop(self):
        """Monitor loop that runs in separate thread"""
        while self.is_monitoring:
            time.sleep(30)  # Check every 30 seconds
            
            if not self.is_monitoring:
                break
                
            current_time = time.time()
            time_since_update = current_time - self.last_update
            
            if time_since_update > self.timeout_seconds:
                logger.warning(f"Processing appears stuck! No progress for {time_since_update:.1f}s")
                self._log_system_stats()
            
            # Log periodic progress
            if self.processed_count > 0:
                elapsed = current_time - self.start_time
                rate = self.processed_count / elapsed
                eta = (self.total_count - self.processed_count) / rate if rate > 0 else float('inf')
                logger.info(f"Progress: {self.processed_count}/{self.total_count}, ETA: {eta:.1f}s")
    
    def _log_system_stats(self):
        """Log system resource usage"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            logger.info(f"System stats - CPU: {cpu_percent}%, Memory: {memory.percent}% used")
            logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
            
            # Log process-specific stats
            process = psutil.Process()
            process_memory = process.memory_info()
            logger.info(f"Process memory: {process_memory.rss / (1024**3):.1f} GB RSS")
            
        except Exception as e:
            logger.warning(f"Could not get system stats: {e}")


def timeout_wrapper(timeout_seconds: int = 60):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise TimeoutError(f"Function timed out after {timeout_seconds}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


class RobustDatasetProcessor:
    """Robust dataset processor with monitoring and fallbacks"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 timeout_per_sample: int = 30,
                 max_retries: int = 3):
        self.max_workers = max_workers or min(8, psutil.cpu_count())
        self.timeout_per_sample = timeout_per_sample
        self.max_retries = max_retries
        self.monitor = ProcessingMonitor()
        
    def process_dataset_with_monitoring(self, 
                                      dataset, 
                                      process_func: Callable,
                                      desc: str = "Processing dataset") -> Any:
        """
        Process dataset with comprehensive monitoring and fallbacks
        
        Args:
            dataset: HuggingFace dataset to process
            process_func: Function to apply to each sample
            desc: Description for progress bar
            
        Returns:
            Processed dataset
        """
        logger.info(f"Starting {desc} with {len(dataset)} samples")
        
        # Start monitoring
        self.monitor.start_monitoring(len(dataset))
        
        try:
            # Try parallel processing first
            return self._try_parallel_processing(dataset, process_func, desc)
            
        except Exception as e:
            logger.warning(f"Parallel processing failed: {e}")
            logger.info("Falling back to sequential processing...")
            
            try:
                return self._sequential_processing(dataset, process_func, desc)
            except Exception as e2:
                logger.error(f"Sequential processing also failed: {e2}")
                raise e2
                
        finally:
            self.monitor.stop_monitoring()
    
    def _try_parallel_processing(self, dataset, process_func, desc):
        """Try parallel processing with monitoring"""
        
        @timeout_wrapper(timeout_seconds=self.timeout_per_sample)
        def safe_process_func(sample):
            return process_func(sample)
        
        # Use conservative number of processes
        num_proc = min(self.max_workers, max(1, len(dataset) // 100))
        
        logger.info(f"Attempting parallel processing with {num_proc} processes")
        
        processed_dataset = dataset.map(
            safe_process_func,
            remove_columns=list(dataset.features),
            num_proc=num_proc,
            desc=desc,
            load_from_cache_file=False,
            keep_in_memory=False  # Don't keep in memory to avoid OOM
        )
        
        return processed_dataset
    
    def _sequential_processing(self, dataset, process_func, desc):
        """Sequential processing with progress monitoring"""
        logger.info("Processing samples sequentially...")
        
        processed_samples = []
        failed_samples = 0
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                
                # Process with timeout
                @timeout_wrapper(timeout_seconds=self.timeout_per_sample)
                def process_single():
                    return process_func(sample)
                
                processed_sample = process_single()
                processed_samples.append(processed_sample)
                
                # Update progress
                self.monitor.update_progress(i + 1)
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                failed_samples += 1
                
                # Create dummy sample or skip
                if hasattr(self, '_create_dummy_sample'):
                    dummy_sample = self._create_dummy_sample(i)
                    processed_samples.append(dummy_sample)
                else:
                    # Skip this sample
                    continue
        
        logger.info(f"Sequential processing completed. Failed samples: {failed_samples}")
        
        # Convert back to dataset
        if processed_samples:
            keys = processed_samples[0].keys()
            processed_dict = {k: [sample[k] for sample in processed_samples] for k in keys}
            from datasets import Dataset
            return Dataset.from_dict(processed_dict)
        else:
            raise ValueError("No samples could be processed")


def create_robust_processor(**kwargs) -> RobustDatasetProcessor:
    """Create a robust dataset processor with default settings"""
    return RobustDatasetProcessor(**kwargs)


def validate_dataset_sample(sample: Dict[str, Any]) -> bool:
    """
    Validate a dataset sample to ensure it has required fields
    
    Args:
        sample: Dataset sample to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['pixel_values', 'question_tokenized', 'steps_tokenized', 'answer_tokenized']
    
    for field in required_fields:
        if field not in sample:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Validate tensor shapes
    try:
        pixel_values = sample['pixel_values']
        if not hasattr(pixel_values, 'shape') or len(pixel_values.shape) != 4:
            logger.warning(f"Invalid pixel_values shape: {getattr(pixel_values, 'shape', 'no shape')}")
            return False
            
        # Validate tokenized fields are lists
        for field in ['question_tokenized', 'steps_tokenized', 'answer_tokenized']:
            if not isinstance(sample[field], (list, tuple)):
                logger.warning(f"Field {field} is not a list/tuple")
                return False
                
    except Exception as e:
        logger.warning(f"Error validating sample: {e}")
        return False
    
    return True


def estimate_memory_usage(dataset_size: int, 
                         image_size: int = 448, 
                         max_patches: int = 12,
                         avg_text_length: int = 512) -> Dict[str, float]:
    """
    Estimate memory usage for dataset processing
    
    Args:
        dataset_size: Number of samples in dataset
        image_size: Image patch size
        max_patches: Maximum patches per image
        avg_text_length: Average text sequence length
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Image memory: patches * channels * height * width * 4 bytes (float32)
    image_memory_per_sample = max_patches * 3 * image_size * image_size * 4
    
    # Text memory: sequence length * 4 bytes (int32)
    text_memory_per_sample = avg_text_length * 4
    
    total_per_sample = image_memory_per_sample + text_memory_per_sample
    total_dataset_memory = total_per_sample * dataset_size
    
    return {
        'per_sample_mb': total_per_sample / (1024 * 1024),
        'total_dataset_gb': total_dataset_memory / (1024 ** 3),
        'image_memory_gb': (image_memory_per_sample * dataset_size) / (1024 ** 3),
        'text_memory_gb': (text_memory_per_sample * dataset_size) / (1024 ** 3)
    }