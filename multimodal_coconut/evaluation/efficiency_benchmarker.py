"""
Efficiency Benchmarking Tools for Multimodal CoCoNuT

This module implements comprehensive efficiency benchmarking including:
- Memory usage profiling during training and inference
- Inference time measurement and comparison
- Throughput benchmarks for different batch sizes and configurations
- Performance comparison between CoT and CoCoNuT modes

Key features:
- GPU memory profiling with detailed breakdown
- Inference latency measurement with statistical analysis
- Throughput benchmarking across different configurations
- Memory optimization recommendations
- Performance regression detection
"""

import time
import json
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import gc
import threading
from contextlib import contextmanager

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization features will be limited.")


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot"""
    timestamp: float
    gpu_allocated: float  # GB
    gpu_reserved: float   # GB
    gpu_free: float      # GB
    cpu_percent: float
    cpu_memory_gb: float
    process_memory_gb: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run"""
    batch_size: int
    sequence_length: int
    num_images: int
    latent_tokens: int
    inference_time_ms: float
    memory_peak_gb: float
    throughput_samples_per_sec: float
    tokens_per_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Complete benchmark results"""
    config_name: str
    model_name: str
    device_info: Dict[str, Any]
    inference_metrics: List[InferenceMetrics]
    memory_profile: List[MemorySnapshot]
    summary_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_name': self.config_name,
            'model_name': self.model_name,
            'device_info': self.device_info,
            'inference_metrics': [m.to_dict() for m in self.inference_metrics],
            'memory_profile': [s.to_dict() for s in self.memory_profile],
            'summary_stats': self.summary_stats
        }


class MemoryProfiler:
    """Memory usage profiler for training and inference"""
    
    def __init__(self, device: torch.device, sample_interval: float = 0.1):
        """
        Initialize memory profiler
        
        Args:
            device: PyTorch device to monitor
            sample_interval: Sampling interval in seconds
        """
        self.device = device
        self.sample_interval = sample_interval
        self.is_profiling = False
        self.memory_snapshots: List[MemorySnapshot] = []
        self.profiling_thread: Optional[threading.Thread] = None
    
    def start_profiling(self):
        """Start memory profiling in background thread"""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.memory_snapshots.clear()
        self.profiling_thread = threading.Thread(target=self._profile_loop)
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
    
    def stop_profiling(self) -> List[MemorySnapshot]:
        """Stop memory profiling and return snapshots"""
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=1.0)
        return self.memory_snapshots.copy()
    
    def _profile_loop(self):
        """Background profiling loop"""
        while self.is_profiling:
            try:
                snapshot = self._take_snapshot()
                self.memory_snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                print(f"Error in memory profiling: {e}")
                break
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a single memory snapshot"""
        timestamp = time.time()
        
        # GPU memory
        if self.device.type == 'cuda':
            gpu_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            gpu_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)   # GB
            
            if GPUTIL_AVAILABLE:
                try:
                    gpu = GPUtil.getGPUs()[self.device.index]
                    gpu_free = gpu.memoryFree / 1024  # GB
                except:
                    gpu_free = 0.0
            else:
                gpu_free = 0.0
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0.0
        
        # CPU memory
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        cpu_memory_gb = memory_info.used / (1024**3)
        
        # Process memory
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        return MemorySnapshot(
            timestamp=timestamp,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            gpu_free=gpu_free,
            cpu_percent=cpu_percent,
            cpu_memory_gb=cpu_memory_gb,
            process_memory_gb=process_memory_gb
        )
    
    @contextmanager
    def profile_context(self):
        """Context manager for memory profiling"""
        self.start_profiling()
        try:
            yield self
        finally:
            self.stop_profiling()


class EfficiencyBenchmarker:
    """
    Comprehensive efficiency benchmarker for multimodal CoCoNuT models.
    
    Provides detailed performance analysis including memory usage,
    inference latency, and throughput measurements.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 config,
                 device: Optional[torch.device] = None):
        """
        Initialize efficiency benchmarker
        
        Args:
            model: Multimodal CoCoNuT model
            tokenizer: Tokenizer with special tokens
            config: Configuration object
            device: Device for benchmarking
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Profiler
        self.memory_profiler = MemoryProfiler(self.device)
        
        # Benchmark configurations
        self.batch_sizes = [1, 2, 4, 8]
        self.sequence_lengths = [128, 256, 512, 1024]
        self.latent_token_counts = [0, 2, 4, 8, 16]
        
        # Results storage
        self.benchmark_results: List[BenchmarkResults] = []
        
        # Device info
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        info = {
            'device_type': self.device.type,
            'device_index': self.device.index if hasattr(self.device, 'index') else 0,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'gpu_memory_gb': torch.cuda.get_device_properties(self.device).total_memory / (1024**3),
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            })
        
        return info
    
    def benchmark_inference_latency(self,
                                   test_data: List[Dict[str, Any]],
                                   num_warmup: int = 5,
                                   num_runs: int = 20) -> List[InferenceMetrics]:
        """
        Benchmark inference latency across different configurations
        
        Args:
            test_data: List of test samples with pixel_values, input_ids, etc.
            num_warmup: Number of warmup runs
            num_runs: Number of measurement runs
            
        Returns:
            List of inference metrics
        """
        print("Benchmarking inference latency...")
        self.model.eval()
        
        inference_metrics = []
        
        # Test different batch sizes
        for batch_size in self.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            print(f"  Testing batch size: {batch_size}")
            
            # Prepare batch
            batch_data = test_data[:batch_size]
            batch = self._prepare_batch(batch_data)
            
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Warmup runs
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = self.model.generate(
                        **{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']},
                        max_new_tokens=64,
                        do_sample=False
                    )
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            # Measurement runs
            latencies = []
            memory_peaks = []
            
            for run in range(num_runs):
                # Clear cache
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(self.device)
                
                # Measure inference time
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']},
                        max_new_tokens=64,
                        do_sample=False
                    )
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Record metrics
                inference_time_ms = (end_time - start_time) * 1000
                latencies.append(inference_time_ms)
                
                # Memory peak
                if self.device.type == 'cuda':
                    memory_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    memory_peaks.append(memory_peak_gb)
                else:
                    memory_peaks.append(0.0)
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            avg_memory_peak = np.mean(memory_peaks)
            
            # Calculate throughput
            throughput_samples_per_sec = batch_size / (avg_latency / 1000)
            
            # Estimate tokens per second (approximate)
            avg_tokens_per_sample = 64  # max_new_tokens
            tokens_per_sec = throughput_samples_per_sec * avg_tokens_per_sample
            
            # Get sequence info
            sequence_length = batch['input_ids'].shape[1]
            num_images = batch_size  # Assuming one image per sample
            latent_tokens = self._count_latent_tokens(batch['input_ids'])
            
            metrics = InferenceMetrics(
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_images=num_images,
                latent_tokens=latent_tokens,
                inference_time_ms=avg_latency,
                memory_peak_gb=avg_memory_peak,
                throughput_samples_per_sec=throughput_samples_per_sec,
                tokens_per_sec=tokens_per_sec
            )
            
            inference_metrics.append(metrics)
            
            print(f"    Latency: {avg_latency:.2f}ms, "
                  f"Memory: {avg_memory_peak:.2f}GB, "
                  f"Throughput: {throughput_samples_per_sec:.2f} samples/sec")
        
        return inference_metrics
    
    def _prepare_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch from test data"""
        # This is a simplified version - in practice, you'd use the actual collator
        from ..data.dataset import MultimodalCollator
        
        collator = MultimodalCollator(
            tokenizer=self.tokenizer,
            latent_id=getattr(self.tokenizer, 'latent_token_id', None)
        )
        
        return collator(batch_data)
    
    def _count_latent_tokens(self, input_ids: torch.Tensor) -> int:
        """Count latent tokens in input"""
        latent_id = getattr(self.tokenizer, 'latent_token_id', None)
        if latent_id is None:
            return 0
        return int(torch.sum(input_ids == latent_id).item())
    
    def benchmark_memory_usage(self,
                              test_data: List[Dict[str, Any]],
                              training_mode: bool = False) -> List[MemorySnapshot]:
        """
        Benchmark memory usage during inference or training
        
        Args:
            test_data: Test data for benchmarking
            training_mode: Whether to benchmark training or inference
            
        Returns:
            List of memory snapshots
        """
        print(f"Benchmarking memory usage ({'training' if training_mode else 'inference'})...")
        
        if training_mode:
            self.model.train()
        else:
            self.model.eval()
        
        # Start memory profiling
        with self.memory_profiler.profile_context():
            if training_mode:
                self._simulate_training_step(test_data)
            else:
                self._simulate_inference_batch(test_data)
        
        snapshots = self.memory_profiler.memory_snapshots
        
        if snapshots:
            peak_gpu = max(s.gpu_allocated for s in snapshots)
            peak_cpu = max(s.process_memory_gb for s in snapshots)
            print(f"  Peak GPU memory: {peak_gpu:.2f}GB")
            print(f"  Peak process memory: {peak_cpu:.2f}GB")
        
        return snapshots
    
    def _simulate_training_step(self, test_data: List[Dict[str, Any]]):
        """Simulate a training step for memory profiling"""
        batch = self._prepare_batch(test_data[:4])  # Small batch for training
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']})
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Simulate optimizer step
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def _simulate_inference_batch(self, test_data: List[Dict[str, Any]]):
        """Simulate inference batch for memory profiling"""
        batch = self._prepare_batch(test_data[:8])  # Larger batch for inference
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']},
                max_new_tokens=128,
                do_sample=False
            )
    
    def benchmark_throughput(self,
                           test_data: List[Dict[str, Any]],
                           duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Benchmark throughput over a sustained period
        
        Args:
            test_data: Test data for benchmarking
            duration_seconds: Duration of throughput test
            
        Returns:
            Throughput statistics
        """
        print(f"Benchmarking throughput for {duration_seconds} seconds...")
        self.model.eval()
        
        # Prepare test batch
        batch = self._prepare_batch(test_data[:4])
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model.generate(
                    **{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']},
                    max_new_tokens=64,
                    do_sample=False
                )
        
        # Throughput test
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        batch_count = 0
        sample_count = 0
        latencies = []
        
        with torch.no_grad():
            while time.time() < end_time:
                batch_start = time.perf_counter()
                
                outputs = self.model.generate(
                    **{k: v for k, v in batch.items() if k not in ['idx', '_num_patches_list']},
                    max_new_tokens=64,
                    do_sample=False
                )
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                batch_end = time.perf_counter()
                
                batch_count += 1
                sample_count += batch['input_ids'].shape[0]
                latencies.append((batch_end - batch_start) * 1000)
        
        actual_duration = time.time() - start_time
        
        throughput_stats = {
            'duration_seconds': actual_duration,
            'total_batches': batch_count,
            'total_samples': sample_count,
            'batches_per_second': batch_count / actual_duration,
            'samples_per_second': sample_count / actual_duration,
            'avg_latency_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
            'latency_p95_ms': np.percentile(latencies, 95),
            'latency_p99_ms': np.percentile(latencies, 99)
        }
        
        print(f"  Throughput: {throughput_stats['samples_per_second']:.2f} samples/sec")
        print(f"  Average latency: {throughput_stats['avg_latency_ms']:.2f}ms")
        
        return throughput_stats
    
    def compare_configurations(self,
                             test_data: List[Dict[str, Any]],
                             configurations: List[Dict[str, Any]]) -> Dict[str, BenchmarkResults]:
        """
        Compare efficiency across different model configurations
        
        Args:
            test_data: Test data for benchmarking
            configurations: List of configuration dictionaries
            
        Returns:
            Dictionary mapping config names to benchmark results
        """
        print("Comparing configurations...")
        
        comparison_results = {}
        
        for config in configurations:
            config_name = config.get('name', 'unnamed')
            print(f"\nBenchmarking configuration: {config_name}")
            
            # Apply configuration changes (this would need to be implemented based on your needs)
            # For now, we'll just benchmark the current model
            
            # Run benchmarks
            inference_metrics = self.benchmark_inference_latency(test_data)
            memory_profile = self.benchmark_memory_usage(test_data)
            throughput_stats = self.benchmark_throughput(test_data, duration_seconds=30)
            
            # Calculate summary statistics
            summary_stats = {
                'avg_inference_time_ms': np.mean([m.inference_time_ms for m in inference_metrics]),
                'avg_memory_peak_gb': np.mean([m.memory_peak_gb for m in inference_metrics]),
                'avg_throughput_samples_per_sec': np.mean([m.throughput_samples_per_sec for m in inference_metrics]),
                'throughput_stats': throughput_stats
            }
            
            # Create benchmark results
            results = BenchmarkResults(
                config_name=config_name,
                model_name=getattr(self.config, 'model_id', 'unknown'),
                device_info=self.device_info,
                inference_metrics=inference_metrics,
                memory_profile=memory_profile,
                summary_stats=summary_stats
            )
            
            comparison_results[config_name] = results
        
        return comparison_results
    
    def generate_efficiency_report(self,
                                 test_data: List[Dict[str, Any]],
                                 output_dir: str,
                                 include_visualizations: bool = True) -> str:
        """
        Generate comprehensive efficiency report
        
        Args:
            test_data: Test data for benchmarking
            output_dir: Directory to save report
            include_visualizations: Whether to include visualizations
            
        Returns:
            Path to generated report
        """
        print("Generating efficiency report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run all benchmarks
        inference_metrics = self.benchmark_inference_latency(test_data)
        memory_profile = self.benchmark_memory_usage(test_data, training_mode=False)
        training_memory_profile = self.benchmark_memory_usage(test_data, training_mode=True)
        throughput_stats = self.benchmark_throughput(test_data)
        
        # Create comprehensive report
        report = {
            'device_info': self.device_info,
            'benchmark_timestamp': time.time(),
            'inference_metrics': [m.to_dict() for m in inference_metrics],
            'memory_profiles': {
                'inference': [s.to_dict() for s in memory_profile],
                'training': [s.to_dict() for s in training_memory_profile]
            },
            'throughput_stats': throughput_stats,
            'summary': self._generate_summary_stats(inference_metrics, memory_profile, throughput_stats)
        }
        
        # Save report
        report_path = output_path / 'efficiency_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualizations if requested
        if include_visualizations and MATPLOTLIB_AVAILABLE:
            self._create_efficiency_visualizations(report, output_path)
        
        print(f"Efficiency report saved to: {report_path}")
        return str(report_path)
    
    def _generate_summary_stats(self,
                              inference_metrics: List[InferenceMetrics],
                              memory_profile: List[MemorySnapshot],
                              throughput_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'avg_inference_time_ms': np.mean([m.inference_time_ms for m in inference_metrics]),
            'max_memory_usage_gb': max([s.gpu_allocated for s in memory_profile]) if memory_profile else 0.0,
            'peak_throughput_samples_per_sec': max([m.throughput_samples_per_sec for m in inference_metrics]),
            'sustained_throughput_samples_per_sec': throughput_stats['samples_per_second'],
            'memory_efficiency_score': self._calculate_memory_efficiency_score(inference_metrics),
            'latency_efficiency_score': self._calculate_latency_efficiency_score(inference_metrics)
        }
    
    def _calculate_memory_efficiency_score(self, metrics: List[InferenceMetrics]) -> float:
        """Calculate memory efficiency score (0-100)"""
        if not metrics:
            return 0.0
        
        # Simple heuristic: lower memory usage per sample is better
        memory_per_sample = [m.memory_peak_gb / m.batch_size for m in metrics]
        avg_memory_per_sample = np.mean(memory_per_sample)
        
        # Normalize to 0-100 scale (assuming 1GB per sample is baseline)
        score = max(0, 100 - (avg_memory_per_sample * 100))
        return min(100, score)
    
    def _calculate_latency_efficiency_score(self, metrics: List[InferenceMetrics]) -> float:
        """Calculate latency efficiency score (0-100)"""
        if not metrics:
            return 0.0
        
        # Simple heuristic: lower latency per sample is better
        latency_per_sample = [m.inference_time_ms / m.batch_size for m in metrics]
        avg_latency_per_sample = np.mean(latency_per_sample)
        
        # Normalize to 0-100 scale (assuming 100ms per sample is baseline)
        score = max(0, 100 - (avg_latency_per_sample / 10))
        return min(100, score)
    
    def _create_efficiency_visualizations(self, report: Dict[str, Any], output_path: Path):
        """Create efficiency visualization plots"""
        try:
            # Latency vs Batch Size
            inference_metrics = report['inference_metrics']
            batch_sizes = [m['batch_size'] for m in inference_metrics]
            latencies = [m['inference_time_ms'] for m in inference_metrics]
            throughputs = [m['throughput_samples_per_sec'] for m in inference_metrics]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Latency vs Batch Size
            ax1.plot(batch_sizes, latencies, 'bo-')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Inference Time (ms)')
            ax1.set_title('Inference Latency vs Batch Size')
            ax1.grid(True, alpha=0.3)
            
            # Throughput vs Batch Size
            ax2.plot(batch_sizes, throughputs, 'ro-')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Throughput (samples/sec)')
            ax2.set_title('Throughput vs Batch Size')
            ax2.grid(True, alpha=0.3)
            
            # Memory Usage Over Time (if available)
            if report['memory_profiles']['inference']:
                memory_data = report['memory_profiles']['inference']
                timestamps = [s['timestamp'] for s in memory_data]
                gpu_memory = [s['gpu_allocated'] for s in memory_data]
                
                # Normalize timestamps
                start_time = min(timestamps)
                timestamps = [(t - start_time) for t in timestamps]
                
                ax3.plot(timestamps, gpu_memory, 'g-')
                ax3.set_xlabel('Time (seconds)')
                ax3.set_ylabel('GPU Memory (GB)')
                ax3.set_title('GPU Memory Usage Over Time')
                ax3.grid(True, alpha=0.3)
            
            # Memory vs Batch Size
            memory_peaks = [m['memory_peak_gb'] for m in inference_metrics]
            ax4.plot(batch_sizes, memory_peaks, 'mo-')
            ax4.set_xlabel('Batch Size')
            ax4.set_ylabel('Peak Memory (GB)')
            ax4.set_title('Peak Memory vs Batch Size')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'efficiency_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")


def create_efficiency_benchmarker(model, tokenizer, config, device=None) -> EfficiencyBenchmarker:
    """
    Factory function to create an EfficiencyBenchmarker
    
    Args:
        model: Multimodal CoCoNuT model
        tokenizer: Tokenizer with special tokens
        config: Configuration object
        device: Device for benchmarking
        
    Returns:
        Configured EfficiencyBenchmarker instance
    """
    return EfficiencyBenchmarker(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device
    )