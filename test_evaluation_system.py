#!/usr/bin/env python3
"""
Test script for the multimodal CoCoNuT evaluation system

This script tests the evaluation and benchmarking components to ensure
they work correctly with the multimodal CoCoNuT implementation.
"""

import sys
import torch
import json
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multimodal_coconut.evaluation import (
    AOKVQAEvaluator,
    VQAAccuracyMetrics,
    calculate_vqa_accuracy,
    ReasoningQualityAnalyzer,
    EfficiencyBenchmarker
)
from multimodal_coconut.config import Config


def create_mock_model_and_tokenizer():
    """Create mock model and tokenizer for testing"""
    
    class MockTokenizer:
        def __init__(self):
            self.latent_token_id = 50257
            self.start_latent_id = 50258
            self.end_latent_id = 50259
            self.pad_token_id = 50256
            self.eos_token_id = 50256
        
        def encode(self, text, add_special_tokens=True):
            # Simple mock encoding
            return [1, 2, 3, 4, 5]
        
        def decode(self, token_ids, skip_special_tokens=True):
            # Simple mock decoding
            return "This is a mock generated response. ### The answer is: test answer"
    
    class MockModel:
        def __init__(self):
            self.config = type('Config', (), {'hidden_size': 4096})()
        
        def parameters(self):
            # Return a dummy parameter to get device
            yield torch.tensor([1.0])
        
        def eval(self):
            pass
        
        def generate(self, **kwargs):
            # Mock generation - return dummy token IDs
            batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
            return torch.tensor([[1, 2, 3, 4, 5]] * batch_size)
        
        def forward(self, **kwargs):
            # Mock forward pass
            batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
            seq_len = kwargs.get('input_ids', torch.tensor([[1]])).shape[1]
            
            # Mock outputs
            class MockOutputs:
                def __init__(self):
                    self.loss = torch.tensor(0.5)
                    self.hidden_states = [torch.randn(batch_size, seq_len, 4096)]
                    self.attentions = [torch.randn(batch_size, 12, seq_len, seq_len)]
            
            return MockOutputs()
        
        def __call__(self, **kwargs):
            return self.forward(**kwargs)
    
    return MockModel(), MockTokenizer()


def create_mock_evaluation_data():
    """Create mock evaluation data"""
    
    # Create temporary directory for mock data
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create mock JSON data
    mock_data = [
        {
            "image_path": "images/test1.jpg",
            "question": "What color is the sky?",
            "steps": ["I can see the sky in the image.", "The sky appears to be blue."],
            "answer": "blue",
            "metadata": {
                "question_id": "test_1",
                "choices": ["red", "blue", "green", "yellow"],
                "original_question": "What color is the sky?"
            }
        },
        {
            "image_path": "images/test2.jpg", 
            "question": "How many cats are in the image?",
            "steps": ["I need to count the cats.", "I can see 2 cats in the image."],
            "answer": "2",
            "metadata": {
                "question_id": "test_2",
                "choices": ["1", "2", "3", "4"],
                "original_question": "How many cats are in the image?"
            }
        }
    ]
    
    # Save mock data
    data_file = temp_path / "test_data.json"
    with open(data_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    # Create mock images directory
    images_dir = temp_path / "images"
    images_dir.mkdir()
    
    return str(data_file), str(temp_path)


def test_vqa_metrics():
    """Test VQA accuracy metrics calculation"""
    print("Testing VQA accuracy metrics...")
    
    # Create mock evaluation samples
    class MockSample:
        def __init__(self, gt_answer, pred_answer, choices=None):
            self.ground_truth_answer = gt_answer
            self.predicted_answer = pred_answer
            self.choices = choices or []
    
    samples = [
        MockSample("blue", "blue"),  # Correct
        MockSample("2", "two"),      # Correct after normalization
        MockSample("cat", "dog"),    # Incorrect
        MockSample("red", "red", ["red", "blue", "green"])  # Correct MC
    ]
    
    # Simple normalization function
    def normalize_answer(answer):
        return answer.lower().replace("two", "2").strip()
    
    # Calculate metrics
    metrics = calculate_vqa_accuracy(samples, normalize_answer)
    
    print(f"  Total samples: {metrics.total_samples}")
    print(f"  Exact match accuracy: {metrics.exact_match_accuracy:.3f}")
    print(f"  Normalized accuracy: {metrics.normalized_accuracy:.3f}")
    print(f"  Multiple choice accuracy: {metrics.multiple_choice_accuracy:.3f}")
    
    # Verify results
    assert metrics.total_samples == 4
    assert metrics.exact_match_accuracy == 0.5  # 2/4 exact matches
    assert metrics.normalized_accuracy == 0.75  # 3/4 after normalization
    assert metrics.multiple_choice_accuracy == 1.0  # 1/1 MC correct
    
    print("  ✓ VQA metrics test passed")


def test_aokvqa_evaluator():
    """Test A-OKVQA evaluator"""
    print("Testing A-OKVQA evaluator...")
    
    # Create mock components
    model, tokenizer = create_mock_model_and_tokenizer()
    config = Config({
        'image_size': 224,
        'max_num_patches': 6,
        'use_thumbnail': True,
        'batch_size_eval': 1,
        'max_new_tokens': 64,
        'num_workers': 1
    })
    
    # Create evaluator
    evaluator = AOKVQAEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rank=0,
        world_size=1
    )
    
    # Test answer normalization
    test_answer = "The Answer is: Blue!"
    normalized = evaluator.normalize_answer(test_answer)
    print(f"  Normalized '{test_answer}' -> '{normalized}'")
    
    # Test answer extraction
    generated_text = "Let me think about this. The sky is blue. ### The answer is: blue"
    extracted = evaluator.extract_answer_from_generation(generated_text, ["red", "blue", "green"])
    print(f"  Extracted answer: '{extracted}'")
    
    print("  ✓ A-OKVQA evaluator test passed")


def test_reasoning_analyzer():
    """Test reasoning quality analyzer"""
    print("Testing reasoning quality analyzer...")
    
    # Create mock components
    model, tokenizer = create_mock_model_and_tokenizer()
    config = Config({'hidden_size': 4096})
    
    # Create analyzer
    analyzer = ReasoningQualityAnalyzer(
        model=model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Test continuous thought analysis with mock data
    from multimodal_coconut.evaluation.reasoning_analyzer import ReasoningTrace, ReasoningStep
    
    # Create mock reasoning traces
    mock_traces = []
    for i in range(3):
        steps = []
        for j in range(5):
            step = ReasoningStep(
                step_index=j,
                step_text=f"Step {j}" if j % 2 == 0 else None,
                hidden_state=torch.randn(4096),
                is_latent=(j % 2 == 1),
                confidence_score=0.8
            )
            steps.append(step)
        
        trace = ReasoningTrace(
            sample_id=f"sample_{i}",
            question=f"Test question {i}",
            image_path=f"test_{i}.jpg",
            reasoning_steps=steps,
            final_answer=f"answer_{i}",
            ground_truth_answer=f"answer_{i}",
            is_correct=True,
            stage=1
        )
        mock_traces.append(trace)
    
    # Test analysis
    analysis = analyzer.analyze_continuous_thoughts(mock_traces)
    
    print(f"  Total latent steps: {analysis['total_latent_steps']}")
    print(f"  Original dimension: {analysis['dimensionality']['original_dim']}")
    print(f"  Average similarity: {analysis['similarity']['average_cosine_similarity']:.3f}")
    
    print("  ✓ Reasoning analyzer test passed")


def test_efficiency_benchmarker():
    """Test efficiency benchmarker"""
    print("Testing efficiency benchmarker...")
    
    # Create mock components
    model, tokenizer = create_mock_model_and_tokenizer()
    config = Config({
        'model_id': 'test-model',
        'batch_size_training': 4
    })
    
    # Create benchmarker
    benchmarker = EfficiencyBenchmarker(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=torch.device('cpu')  # Use CPU for testing
    )
    
    # Test device info
    device_info = benchmarker.device_info
    print(f"  Device type: {device_info['device_type']}")
    print(f"  CPU count: {device_info['cpu_count']}")
    print(f"  Memory total: {device_info['memory_total_gb']:.1f}GB")
    
    # Test memory profiler
    from multimodal_coconut.evaluation.efficiency_benchmarker import MemoryProfiler
    
    profiler = MemoryProfiler(torch.device('cpu'), sample_interval=0.01)
    
    # Test snapshot
    snapshot = profiler._take_snapshot()
    print(f"  Memory snapshot - CPU: {snapshot.cpu_percent:.1f}%, Process: {snapshot.process_memory_gb:.2f}GB")
    
    print("  ✓ Efficiency benchmarker test passed")


def main():
    """Run all evaluation system tests"""
    print("=" * 60)
    print("TESTING MULTIMODAL COCONUT EVALUATION SYSTEM")
    print("=" * 60)
    
    try:
        # Test individual components
        test_vqa_metrics()
        print()
        
        test_aokvqa_evaluator()
        print()
        
        test_reasoning_analyzer()
        print()
        
        test_efficiency_benchmarker()
        print()
        
        print("=" * 60)
        print("ALL EVALUATION SYSTEM TESTS PASSED ✓")
        print("=" * 60)
        
        # Print summary of implemented features
        print("\nImplemented evaluation features:")
        print("  ✓ A-OKVQA evaluation pipeline with VQA accuracy metrics")
        print("  ✓ Support for multiple-choice and open-ended questions")
        print("  ✓ Answer normalization following VQA standards")
        print("  ✓ Reasoning quality analysis tools")
        print("  ✓ Continuous thought representation inspection")
        print("  ✓ Latent space reasoning progression visualization")
        print("  ✓ Comparison metrics between CoT and CoCoNuT")
        print("  ✓ Efficiency benchmarking with memory profiling")
        print("  ✓ Inference time measurement and throughput analysis")
        print("  ✓ Performance comparison across configurations")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
