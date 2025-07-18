"""
VQA Accuracy Metrics for Multimodal CoCoNuT Evaluation

This module implements standard VQA accuracy metrics calculation,
supporting both exact match and normalized accuracy evaluation.

Key features:
- Exact match accuracy calculation
- Normalized answer accuracy (following VQA evaluation standards)
- Multiple-choice question accuracy
- Support for different answer normalization strategies
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Union
from collections import Counter


@dataclass
class VQAAccuracyMetrics:
    """Container for VQA accuracy metrics"""
    total_samples: int
    exact_match_accuracy: float
    normalized_accuracy: float
    multiple_choice_accuracy: Optional[float] = None
    correct_samples: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_samples': self.total_samples,
            'exact_match_accuracy': self.exact_match_accuracy,
            'normalized_accuracy': self.normalized_accuracy,
            'multiple_choice_accuracy': self.multiple_choice_accuracy,
            'correct_samples': self.correct_samples
        }
    
    def __str__(self) -> str:
        """String representation of metrics"""
        lines = [
            f"VQA Accuracy Metrics:",
            f"  Total samples: {self.total_samples}",
            f"  Exact match accuracy: {self.exact_match_accuracy:.4f}",
            f"  Normalized accuracy: {self.normalized_accuracy:.4f}"
        ]
        
        if self.multiple_choice_accuracy is not None:
            lines.append(f"  Multiple choice accuracy: {self.multiple_choice_accuracy:.4f}")
        
        if self.correct_samples is not None:
            lines.append(f"  Correct samples: {self.correct_samples}")
        
        return '\n'.join(lines)


def calculate_vqa_accuracy(evaluation_samples: List,
                          normalize_fn: Optional[Callable[[str], str]] = None) -> VQAAccuracyMetrics:
    """
    Calculate VQA accuracy metrics from evaluation samples
    
    Args:
        evaluation_samples: List of evaluation samples with ground_truth_answer and predicted_answer
        normalize_fn: Function to normalize answers (optional)
        
    Returns:
        VQA accuracy metrics
    """
    if not evaluation_samples:
        return VQAAccuracyMetrics(0, 0.0, 0.0, None, 0)
    
    total_samples = len(evaluation_samples)
    exact_matches = 0
    normalized_matches = 0
    mc_correct = 0
    mc_total = 0
    
    for sample in evaluation_samples:
        gt_answer = sample.ground_truth_answer
        pred_answer = sample.predicted_answer
        
        # Exact match accuracy
        if gt_answer == pred_answer:
            exact_matches += 1
        
        # Normalized accuracy
        if normalize_fn:
            gt_normalized = normalize_fn(gt_answer)
            pred_normalized = normalize_fn(pred_answer)
            
            if gt_normalized == pred_normalized:
                normalized_matches += 1
        else:
            # Fallback to simple lowercase comparison
            if gt_answer.lower().strip() == pred_answer.lower().strip():
                normalized_matches += 1
        
        # Multiple choice accuracy (if choices are available)
        if hasattr(sample, 'choices') and sample.choices:
            mc_total += 1
            # Check if predicted answer matches any choice exactly
            if pred_answer in sample.choices:
                mc_correct += 1
            # Also check normalized versions
            elif normalize_fn:
                pred_norm = normalize_fn(pred_answer)
                for choice in sample.choices:
                    if normalize_fn(choice) == pred_norm:
                        mc_correct += 1
                        break
    
    # Calculate accuracies
    exact_match_accuracy = exact_matches / total_samples
    normalized_accuracy = normalized_matches / total_samples
    multiple_choice_accuracy = mc_correct / mc_total if mc_total > 0 else None
    
    return VQAAccuracyMetrics(
        total_samples=total_samples,
        exact_match_accuracy=exact_match_accuracy,
        normalized_accuracy=normalized_accuracy,
        multiple_choice_accuracy=multiple_choice_accuracy,
        correct_samples=normalized_matches
    )


def calculate_answer_distribution(evaluation_samples: List) -> Dict[str, Dict[str, int]]:
    """
    Calculate answer distribution statistics
    
    Args:
        evaluation_samples: List of evaluation samples
        
    Returns:
        Dictionary with answer distribution statistics
    """
    gt_answers = [sample.ground_truth_answer for sample in evaluation_samples]
    pred_answers = [sample.predicted_answer for sample in evaluation_samples]
    
    gt_counter = Counter(gt_answers)
    pred_counter = Counter(pred_answers)
    
    return {
        'ground_truth_distribution': dict(gt_counter.most_common(20)),
        'predicted_distribution': dict(pred_counter.most_common(20)),
        'unique_gt_answers': len(gt_counter),
        'unique_pred_answers': len(pred_counter)
    }


def calculate_per_question_type_accuracy(evaluation_samples: List,
                                       normalize_fn: Optional[Callable[[str], str]] = None) -> Dict[str, VQAAccuracyMetrics]:
    """
    Calculate accuracy metrics per question type (if available in metadata)
    
    Args:
        evaluation_samples: List of evaluation samples
        normalize_fn: Function to normalize answers
        
    Returns:
        Dictionary mapping question types to accuracy metrics
    """
    # Group samples by question type
    type_groups = {}
    
    for sample in evaluation_samples:
        # Try to extract question type from metadata
        question_type = 'unknown'
        if hasattr(sample, 'metadata') and sample.metadata:
            question_type = sample.metadata.get('question_type', 'unknown')
        elif hasattr(sample, 'choices') and sample.choices:
            question_type = 'multiple_choice'
        else:
            question_type = 'open_ended'
        
        if question_type not in type_groups:
            type_groups[question_type] = []
        type_groups[question_type].append(sample)
    
    # Calculate metrics for each type
    type_metrics = {}
    for question_type, samples in type_groups.items():
        type_metrics[question_type] = calculate_vqa_accuracy(samples, normalize_fn)
    
    return type_metrics


def calculate_confidence_based_accuracy(evaluation_samples: List,
                                      confidence_scores: Optional[List[float]] = None,
                                      normalize_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """
    Calculate accuracy metrics based on confidence scores (if available)
    
    Args:
        evaluation_samples: List of evaluation samples
        confidence_scores: List of confidence scores for each sample (optional)
        normalize_fn: Function to normalize answers
        
    Returns:
        Dictionary with confidence-based accuracy statistics
    """
    if not confidence_scores or len(confidence_scores) != len(evaluation_samples):
        return {'error': 'Confidence scores not available or mismatched length'}
    
    # Sort samples by confidence
    samples_with_conf = list(zip(evaluation_samples, confidence_scores))
    samples_with_conf.sort(key=lambda x: x[1], reverse=True)  # High confidence first
    
    # Calculate accuracy at different confidence thresholds
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
    threshold_metrics = {}
    
    for threshold in thresholds:
        # Filter samples above threshold
        filtered_samples = [sample for sample, conf in samples_with_conf if conf >= threshold]
        
        if filtered_samples:
            metrics = calculate_vqa_accuracy(filtered_samples, normalize_fn)
            threshold_metrics[f'threshold_{threshold}'] = {
                'accuracy': metrics.normalized_accuracy,
                'num_samples': len(filtered_samples),
                'coverage': len(filtered_samples) / len(evaluation_samples)
            }
    
    return {
        'threshold_metrics': threshold_metrics,
        'total_samples': len(evaluation_samples)
    }


def compare_model_accuracies(model_results: Dict[str, List]) -> Dict[str, Any]:
    """
    Compare accuracy metrics between different models
    
    Args:
        model_results: Dictionary mapping model names to evaluation samples
        
    Returns:
        Comparison statistics
    """
    model_metrics = {}
    
    # Calculate metrics for each model
    for model_name, samples in model_results.items():
        model_metrics[model_name] = calculate_vqa_accuracy(samples)
    
    # Find best performing model
    best_model = max(model_metrics.keys(), 
                    key=lambda x: model_metrics[x].normalized_accuracy)
    
    # Calculate relative improvements
    baseline_acc = min(model_metrics[m].normalized_accuracy for m in model_metrics)
    
    comparison = {
        'model_metrics': {name: metrics.to_dict() for name, metrics in model_metrics.items()},
        'best_model': best_model,
        'best_accuracy': model_metrics[best_model].normalized_accuracy,
        'relative_improvements': {}
    }
    
    for model_name, metrics in model_metrics.items():
        improvement = (metrics.normalized_accuracy - baseline_acc) / baseline_acc * 100
        comparison['relative_improvements'][model_name] = improvement
    
    return comparison