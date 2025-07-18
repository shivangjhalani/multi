#!/usr/bin/env python3
"""
Test script for comprehensive logging and debugging utilities

This script tests:
- W&B integration and experiment tracking
- Enhanced logging with metrics tracking
- Debugging utilities for multimodal training
- Data pipeline debugging
- Model behavior analysis
- Training monitoring
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_coconut.utils.logging import (
    setup_logging,
    get_logger,
    log_metrics,
    MetricsTracker,
    ExperimentTracker,
    MultimodalDebugger,
    create_experiment_tracker,
    create_multimodal_debugger
)

from multimodal_coconut.utils.debug import (
    DataPipelineDebugger,
    ModelBehaviorAnalyzer,
    TrainingMonitor,
    create_comprehensive_debugger
)

from multimodal_coconut.config import Config


def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic logging...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test.log")
        
        # Test logging setup
        logger = setup_logging(
            log_level="INFO",
            log_file=log_file,
            use_wandb=False  # Disable W&B for testing
        )
        
        assert logger is not None
        assert os.path.exists(log_file)
        
        # Test logging messages
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check log file content
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test info message" in content
            assert "Test warning message" in content
            assert "Test error message" in content
    
    print("✓ Basic logging test passed")


def test_metrics_tracker():
    """Test enhanced metrics tracking."""
    print("Testing metrics tracker...")
    
    tracker = MetricsTracker(window_size=10)
    
    # Test metric updates
    for i in range(15):
        tracker.update(
            loss=1.0 / (i + 1),
            accuracy=i / 15.0,
            learning_rate=0.001 * (0.9 ** i)
        )
    
    # Test averages
    averages = tracker.get_averages()
    assert "loss" in averages
    assert "accuracy" in averages
    assert "learning_rate" in averages
    
    # Test recent averages (should only consider last 10 values)
    recent_averages = tracker.get_recent_averages()
    assert len(recent_averages) == 3
    
    # Test statistics
    stats = tracker.get_statistics()
    assert "loss" in stats
    assert "mean" in stats["loss"]
    assert "std" in stats["loss"]
    assert "min" in stats["loss"]
    assert "max" in stats["loss"]
    
    # Test reset
    tracker.reset()
    assert len(tracker.get_averages()) == 0
    
    print("✓ Metrics tracker test passed")


def test_experiment_tracker():
    """Test experiment tracking with W&B integration."""
    print("Testing experiment tracker...")
    
    config = {
        "model_id": "test-model",
        "learning_rate": 0.001,
        "batch_size": 32,
        "wandb_project": "test-project"
    }
    
    # Mock W&B to avoid actual initialization
    with patch('multimodal_coconut.utils.logging.wandb') as mock_wandb:
        mock_run = Mock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.run = mock_run
        
        tracker = ExperimentTracker(
            project_name="test-project",
            experiment_name="test-experiment",
            tags=["test", "debug"],
            config=config
        )
        
        # Test model architecture logging
        mock_model = Mock()
        mock_model.parameters.return_value = [
            Mock(numel=lambda: 1000, requires_grad=True),
            Mock(numel=lambda: 500, requires_grad=False)
        ]
        mock_model.__str__ = Mock(return_value="Test Model Architecture")
        
        tracker.log_model_architecture(mock_model)
        
        # Test training metrics logging
        metrics = {
            "train_loss": 0.5,
            "train_accuracy": 0.8,
            "learning_rate": 0.001
        }
        tracker.log_training_metrics(metrics, step=100, epoch=1, stage=0)
        
        # Test validation results logging
        val_results = {
            "val_loss": 0.4,
            "val_accuracy": 0.85,
            "predictions": [{"input": "test", "output": "result"}]
        }
        tracker.log_validation_results(val_results, step=100)
        
        # Test stage transition logging
        tracker.log_stage_transition(from_stage=0, to_stage=1, epoch=5)
        
        # Verify W&B calls were made
        assert mock_wandb.init.called
        assert mock_run.log.called
        
        tracker.finish()
    
    print("✓ Experiment tracker test passed")


def test_multimodal_debugger():
    """Test multimodal debugging utilities."""
    print("Testing multimodal debugger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        debugger = MultimodalDebugger(save_dir=temp_dir, max_samples=2)
        
        # Create mock batch
        batch = {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (2, 50)),
            "attention_mask": torch.ones(2, 50),
            "labels": torch.randint(0, 1000, (2, 50))
        }
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.decode = lambda x, **kwargs: f"decoded_text_{hash(str(x)) % 1000}"
        mock_tokenizer.latent_token_id = 999
        
        # Test batch debugging
        debugger.debug_batch(batch, mock_tokenizer, step=100, save_images=False)
        
        # Check if debug files were created
        debug_files = list(Path(temp_dir).glob("*.json"))
        assert len(debug_files) > 0
        
        # Test model outputs debugging
        mock_outputs = Mock()
        mock_outputs.loss = torch.tensor(0.5, requires_grad=True)
        mock_outputs.logits = torch.randn(2, 50, 1000)
        mock_outputs.keys = lambda: ["loss", "logits"]
        
        debugger.debug_model_outputs(mock_outputs, step=100)
        
        # Test gradient debugging
        mock_model = Mock()
        mock_params = [
            Mock(grad=torch.randn(100, 50), data=torch.randn(100, 50), shape=(100, 50)),
            Mock(grad=torch.randn(50), data=torch.randn(50), shape=(50,))
        ]
        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_params[0]),
            ("layer1.bias", mock_params[1])
        ]
        
        debugger.debug_gradients(mock_model, step=100)
        
        # Test memory profiling (only if CUDA is available)
        if torch.cuda.is_available():
            debugger.profile_memory_usage(step=100)
    
    print("✓ Multimodal debugger test passed")


def test_data_pipeline_debugger():
    """Test data pipeline debugging utilities."""
    print("Testing data pipeline debugger...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        debugger = DataPipelineDebugger(save_dir=temp_dir)
        
        # Create mock batch
        batch = {
            "pixel_values": torch.randn(3, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (3, 50)),
            "attention_mask": torch.ones(3, 50)
        }
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 1000
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.latent_token_id = 999
        mock_tokenizer.start_latent_id = 998
        mock_tokenizer.end_latent_id = 997
        
        # Test batch analysis
        analysis = debugger.analyze_batch_composition(batch, batch_idx=0, tokenizer=mock_tokenizer)
        
        assert analysis["batch_size"] == 3
        assert "image_info" in analysis
        assert "text_info" in analysis
        assert "special_tokens" in analysis
        
        # Add more batches for statistics
        for i in range(5):
            batch_i = {
                "pixel_values": torch.randn(2 + i, 3, 224, 224),
                "input_ids": torch.randint(0, 1000, (2 + i, 40 + i * 5))
            }
            debugger.analyze_batch_composition(batch_i, batch_idx=i+1, tokenizer=mock_tokenizer)
        
        # Test statistics visualization
        stats = debugger.visualize_batch_statistics(save_plots=False)  # Don't save plots in test
        assert "total_batches" in stats
        assert stats["total_batches"] == 6
    
    print("✓ Data pipeline debugger test passed")


def test_model_behavior_analyzer():
    """Test model behavior analysis."""
    print("Testing model behavior analyzer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = ModelBehaviorAnalyzer(save_dir=temp_dir)
        
        # Create mock model
        mock_model = Mock()
        mock_params = [
            Mock(grad=torch.randn(100, 50), data=torch.randn(100, 50), shape=(100, 50)),
            Mock(grad=torch.randn(50), data=torch.randn(50), shape=(50,))
        ]
        mock_model.named_parameters.return_value = [
            ("layer1.weight", mock_params[0]),
            ("layer1.bias", mock_params[1])
        ]
        mock_model.named_modules.return_value = [
            ("layer1", Mock(spec=torch.nn.Linear))
        ]
        
        # Test gradient flow analysis
        grad_analysis = analyzer.analyze_gradient_flow(mock_model, step=100)
        assert "layers" in grad_analysis
        assert "summary" in grad_analysis
        assert grad_analysis["summary"]["param_count"] == 2
        
        # Test loss tracking
        for i in range(20):
            loss = 1.0 / (i + 1) + 0.1 * np.random.random()
            analyzer.track_loss_landscape(loss, step=i * 10, additional_metrics={"accuracy": i / 20.0})
        
        assert len(analyzer.loss_history) == 20
    
    print("✓ Model behavior analyzer test passed")


def test_training_monitor():
    """Test training monitoring and alerting."""
    print("Testing training monitor...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        monitor = TrainingMonitor(save_dir=temp_dir, patience=5, min_delta=0.01)
        
        # Simulate training metrics
        for step in range(20):
            metrics = {
                "train_loss": 1.0 - step * 0.05 + 0.01 * np.random.random(),
                "val_loss": 1.2 - step * 0.04 + 0.02 * np.random.random(),
                "accuracy": step * 0.04 + 0.01 * np.random.random()
            }
            monitor.update_metrics(metrics, step=step, epoch=step // 5)
        
        # Test metrics history
        assert len(monitor.metrics_history["train_loss"]) == 20
        assert len(monitor.metrics_history["val_loss"]) == 20
        assert len(monitor.metrics_history["accuracy"]) == 20
        
        # Test recommendations
        recommendations = monitor.get_recommendations()
        assert isinstance(recommendations, list)
        
        # Simulate anomalous values
        anomalous_metrics = {
            "train_loss": 100.0,  # Very high loss
            "val_loss": float('nan'),  # NaN value
            "accuracy": float('inf')  # Inf value
        }
        monitor.update_metrics(anomalous_metrics, step=21)
        
        # Check that alerts were generated
        assert len(monitor.alerts) > 0
        
        # Check for invalid value alerts
        invalid_alerts = [alert for alert in monitor.alerts if alert["type"] == "invalid_value"]
        assert len(invalid_alerts) >= 2  # NaN and Inf
    
    print("✓ Training monitor test passed")


def test_comprehensive_debugger():
    """Test comprehensive debugging suite."""
    print("Testing comprehensive debugger...")
    
    config = {
        "debug_save_dir": "test_debug",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001
    }
    
    debugger_suite = create_comprehensive_debugger(config)
    
    assert "data_pipeline" in debugger_suite
    assert "model_behavior" in debugger_suite
    assert "training_monitor" in debugger_suite
    
    assert isinstance(debugger_suite["data_pipeline"], DataPipelineDebugger)
    assert isinstance(debugger_suite["model_behavior"], ModelBehaviorAnalyzer)
    assert isinstance(debugger_suite["training_monitor"], TrainingMonitor)
    
    # Clean up
    if os.path.exists("test_debug"):
        shutil.rmtree("test_debug")
    
    print("✓ Comprehensive debugger test passed")


def test_config_integration():
    """Test integration with configuration system."""
    print("Testing config integration...")
    
    # Test config creation with logging parameters
    config_dict = {
        "name": "test-experiment",
        "use_wandb": True,
        "wandb_project": "test-project",
        "log_level": "DEBUG",
        "enable_debugging": True,
        "debug_frequency": 100,
        "metrics_window_size": 50
    }
    
    config = Config(config_dict)
    
    # Test experiment tracker creation
    with patch('multimodal_coconut.utils.logging.wandb'):
        tracker = create_experiment_tracker(config.to_dict())
        assert tracker.project_name == "test-project"
        assert tracker.config["name"] == "test-experiment"
    
    # Test debugger creation
    debugger = create_multimodal_debugger(config.to_dict())
    assert debugger.max_samples == 5  # Default value
    
    print("✓ Config integration test passed")


def main():
    """Run all logging and debugging tests."""
    print("=" * 60)
    print("TESTING LOGGING AND DEBUGGING UTILITIES")
    print("=" * 60)
    
    try:
        test_basic_logging()
        test_metrics_tracker()
        test_experiment_tracker()
        test_multimodal_debugger()
        test_data_pipeline_debugger()
        test_model_behavior_analyzer()
        test_training_monitor()
        test_comprehensive_debugger()
        test_config_integration()
        
        print("\n" + "=" * 60)
        print("ALL LOGGING AND DEBUGGING TESTS PASSED! ✅")
        print("=" * 60)
        print("\nFeatures tested:")
        print("✓ Basic logging with file output")
        print("✓ Enhanced metrics tracking with statistics")
        print("✓ W&B experiment tracking integration")
        print("✓ Multimodal debugging utilities")
        print("✓ Data pipeline debugging and visualization")
        print("✓ Model behavior analysis")
        print("✓ Training monitoring and alerting")
        print("✓ Comprehensive debugging suite")
        print("✓ Configuration system integration")
        print("\nThe logging and debugging system is ready for use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)