"""
A-OKVQA Evaluation Pipeline for Multimodal CoCoNuT

This module implements the evaluation pipeline for A-OKVQA dataset,
providing VQA accuracy metrics calculation and support for both
multiple-choice and open-ended question evaluation.

Key features:
- Evaluation dataset loader for A-OKVQA format
- VQA accuracy metrics calculation following standard practices
- Support for multiple-choice and direct-answer evaluation modes
- Integration with multimodal CoCoNuT model generation
- Distributed evaluation support
"""

import json
import re
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
from collections import Counter

from ..data.dataset import get_multimodal_dataset, get_multimodal_question_latent_dataset, MultimodalCollator
from ..config import Config
from .metrics import VQAAccuracyMetrics, calculate_vqa_accuracy


@dataclass
class EvaluationSample:
    """Single evaluation sample with ground truth and prediction"""
    question_id: str
    question: str
    image_path: str
    ground_truth_answer: str
    predicted_answer: str
    choices: Optional[List[str]] = None
    rationales: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AOKVQAEvaluator:
    """
    A-OKVQA evaluation pipeline for multimodal CoCoNuT models.
    
    Supports both multiple-choice and open-ended evaluation modes,
    following standard VQA evaluation practices.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 config: Config,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize A-OKVQA evaluator
        
        Args:
            model: Multimodal CoCoNuT model
            tokenizer: Tokenizer with special tokens
            config: Configuration object
            rank: Process rank for distributed evaluation
            world_size: Total number of processes
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Special token IDs
        self.latent_id = getattr(tokenizer, 'latent_token_id', None)
        self.start_id = getattr(tokenizer, 'start_latent_id', None)
        self.end_id = getattr(tokenizer, 'end_latent_id', None)
        
        # Collator for evaluation data
        self.collator = MultimodalCollator(
            tokenizer=tokenizer,
            latent_id=self.latent_id,
            label_pad_token_id=-100
        )
        
        # Evaluation settings
        self.max_new_tokens = getattr(config, 'max_new_tokens', 128)
        self.use_direct_answer = getattr(config, 'use_direct_answer', False)
        self.batch_size_eval = getattr(config, 'batch_size_eval', 1)
        
        # Answer normalization patterns
        self._setup_answer_normalization()
    
    def _setup_answer_normalization(self):
        """Setup answer normalization patterns for VQA evaluation"""
        # Common answer normalization patterns
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
            "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
            "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
            "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
            "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
            "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
            "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's",
            "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
            "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whered": "where'd", "wheres": "where's",
            "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
            "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
            "whove": "who've", "whyll": "why'll", "whyre": "why're",
            "whys": "why's", "wont": "won't", "wouldve": "would've",
            "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'd": "y'all'd",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
            "yall're": "y'all're", "youll": "you'll", "youre": "you're",
            "youve": "you've"
        }
        
        # Manual map for common VQA answers
        self.manual_map = {
            'none': '0',
            'zero': '0',
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8',
            'nine': '9',
            'ten': '10'
        }
        
        # Articles to remove
        self.articles = ['a', 'an', 'the']
        
        # Period strip
        self.period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
        self.comma_strip = re.compile(r'(\d)(\,)(\d)')
        self.punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
                     '>', '<', '@', '`', ',', '?', '!']
    
    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for VQA evaluation following standard practices
        
        Args:
            answer: Raw answer string
            
        Returns:
            Normalized answer string
        """
        if not isinstance(answer, str):
            answer = str(answer)
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Replace contractions
        for word, replacement in self.contractions.items():
            answer = answer.replace(word, replacement)
        
        # Manual mapping for common answers
        for word, replacement in self.manual_map.items():
            answer = answer.replace(word, replacement)
        
        # Remove articles
        answer_words = answer.split()
        answer_words = [word for word in answer_words if word not in self.articles]
        answer = ' '.join(answer_words)
        
        # Remove punctuation
        for p in self.punct:
            answer = answer.replace(p, '')
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Handle periods and commas
        answer = self.period_strip.sub('', answer)
        answer = self.comma_strip.sub(r'\1\3', answer)
        
        return answer.strip()
    
    def extract_answer_from_generation(self, generated_text: str, choices: Optional[List[str]] = None) -> str:
        """
        Extract answer from model generation
        
        Args:
            generated_text: Full generated text from model
            choices: Multiple choice options (if applicable)
            
        Returns:
            Extracted answer string
        """
        # Remove special tokens and clean up
        generated_text = generated_text.replace('<|start-latent|>', '').replace('<|latent|>', '').replace('<|end-latent|>', '')
        
        # Try to extract answer after "###" marker (following CoCoNuT pattern)
        if "###" in generated_text:
            answer_part = generated_text.split("###")[-1].strip()
        else:
            # Fallback: take the last line or sentence
            lines = generated_text.strip().split('\n')
            answer_part = lines[-1].strip() if lines else generated_text.strip()
        
        # Clean up the answer
        answer_part = answer_part.replace(',', '').strip()
        
        # For multiple choice questions, try to match with choices
        if choices:
            # First try exact match with choices
            for i, choice in enumerate(choices):
                if choice.lower() in answer_part.lower():
                    return choice
            
            # Try to find choice index (0, 1, 2, 3)
            for i, choice in enumerate(choices):
                if str(i) in answer_part:
                    return choice
            
            # Try to find choice letter (A, B, C, D)
            choice_letters = ['A', 'B', 'C', 'D']
            for i, letter in enumerate(choice_letters[:len(choices)]):
                if letter.lower() in answer_part.lower():
                    return choices[i]
        
        return answer_part
    
    def load_evaluation_dataset(self,
                               data_path: str,
                               image_root: str,
                               scheduled_stage: int = 0,
                               max_samples: Optional[int] = None) -> torch.utils.data.DataLoader:
        """
        Load evaluation dataset for A-OKVQA
        
        Args:
            data_path: Path to evaluation data JSON file
            image_root: Root directory for images
            scheduled_stage: Current training stage (for latent token insertion)
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            DataLoader for evaluation
        """
        # Load base dataset
        base_dataset = get_multimodal_dataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            image_root=image_root,
            image_size=getattr(self.config, 'image_size', 448),
            max_num_patches=getattr(self.config, 'max_num_patches', 12),
            use_thumbnail=getattr(self.config, 'use_thumbnail', True),
            max_size=max_samples if max_samples else 1000000000
        )
        
        # Prepare dataset for generation (questions + latent tokens)
        eval_dataset = get_multimodal_question_latent_dataset(
            scheduled_stage=scheduled_stage,
            base_dataset_valid=base_dataset,
            configs=self.config,
            start_id=self.start_id,
            latent_id=self.latent_id,
            end_id=self.end_id,
            no_special_marker=getattr(self.config, 'no_special_marker', False)
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(eval_dataset, shuffle=False) if self.world_size > 1 else None
        
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=getattr(self.config, 'num_workers', 1),
            pin_memory=True,
            collate_fn=self.collator,
            sampler=sampler
        )
        
        return dataloader
    
    def load_ground_truth_data(self, data_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Load ground truth data for evaluation
        
        Args:
            data_path: Path to ground truth data JSON file
            
        Returns:
            Dictionary mapping sample indices to ground truth data
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create mapping from index to ground truth
        ground_truth = {}
        for idx, sample in enumerate(data):
            ground_truth[idx] = {
                'question': sample.get('question', ''),
                'answer': sample.get('answer', ''),
                'choices': sample.get('metadata', {}).get('choices', []),
                'rationales': sample.get('steps', []),
                'question_id': sample.get('metadata', {}).get('question_id', str(idx)),
                'image_path': sample.get('image_path', ''),
                'original_question': sample.get('metadata', {}).get('original_question', sample.get('question', ''))
            }
        
        return ground_truth
    
    def evaluate_batch(self, 
                      batch: Dict[str, torch.Tensor],
                      ground_truth: Dict[int, Dict[str, Any]]) -> List[EvaluationSample]:
        """
        Evaluate a single batch
        
        Args:
            batch: Input batch for model
            ground_truth: Ground truth data mapping
            
        Returns:
            List of evaluation samples with predictions
        """
        # Move batch to device
        device = next(self.model.parameters()).device
        batch_device = {
            key: batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
            for key in batch.keys() if key not in ["idx", "_num_patches_list"]
        }
        
        # Generate responses
        with torch.no_grad():
            # Use synced_gpus for distributed inference
            synced_gpus = self.world_size > 1 and not getattr(self.config, 'only_eval', False)
            
            outputs = self.model.generate(
                **batch_device,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic generation for evaluation
                temperature=1.0,
                synced_gpus=synced_gpus,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Process results
        evaluation_samples = []
        
        for i in range(len(outputs)):
            # Get sample index
            if 'idx' in batch:
                sample_idx = batch['idx'][i].item() if torch.is_tensor(batch['idx'][i]) else batch['idx'][i]
            else:
                sample_idx = i
            
            # Get ground truth
            gt_data = ground_truth.get(sample_idx, {})
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # Extract answer
            predicted_answer = self.extract_answer_from_generation(
                generated_text, 
                gt_data.get('choices', [])
            )
            
            # Create evaluation sample
            eval_sample = EvaluationSample(
                question_id=gt_data.get('question_id', str(sample_idx)),
                question=gt_data.get('question', ''),
                image_path=gt_data.get('image_path', ''),
                ground_truth_answer=gt_data.get('answer', ''),
                predicted_answer=predicted_answer,
                choices=gt_data.get('choices', []),
                rationales=gt_data.get('rationales', []),
                metadata={
                    'generated_text': generated_text,
                    'sample_idx': sample_idx,
                    'original_question': gt_data.get('original_question', '')
                }
            )
            
            evaluation_samples.append(eval_sample)
        
        return evaluation_samples
    
    def evaluate(self,
                data_path: str,
                image_root: str,
                scheduled_stage: int = 0,
                max_samples: Optional[int] = None,
                save_results: bool = True,
                results_path: Optional[str] = None) -> VQAAccuracyMetrics:
        """
        Run complete evaluation on A-OKVQA dataset
        
        Args:
            data_path: Path to evaluation data JSON file
            image_root: Root directory for images
            scheduled_stage: Current training stage
            max_samples: Maximum number of samples to evaluate
            save_results: Whether to save detailed results
            results_path: Path to save results (optional)
            
        Returns:
            VQA accuracy metrics
        """
        if self.rank == 0:
            print("=" * 60)
            print("A-OKVQA EVALUATION")
            print("=" * 60)
            print(f"Data path: {data_path}")
            print(f"Image root: {image_root}")
            print(f"Scheduled stage: {scheduled_stage}")
            print(f"Max samples: {max_samples}")
            print("=" * 60)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Load evaluation dataset
        eval_dataloader = self.load_evaluation_dataset(
            data_path=data_path,
            image_root=image_root,
            scheduled_stage=scheduled_stage,
            max_samples=max_samples
        )
        
        # Load ground truth data
        ground_truth = self.load_ground_truth_data(data_path)
        
        # Collect all evaluation samples
        all_evaluation_samples = []
        
        # Progress bar
        if self.rank == 0:
            pbar = tqdm(
                total=len(eval_dataloader),
                desc="Evaluating A-OKVQA",
                colour="blue",
                dynamic_ncols=True
            )
        
        # Evaluate batches
        for batch_idx, batch in enumerate(eval_dataloader):
            try:
                # Evaluate batch
                batch_samples = self.evaluate_batch(batch, ground_truth)
                all_evaluation_samples.extend(batch_samples)
                
                # Update progress
                if self.rank == 0:
                    pbar.update(1)
                    if len(all_evaluation_samples) > 0:
                        # Calculate running accuracy
                        correct = sum(1 for s in all_evaluation_samples 
                                    if self.normalize_answer(s.predicted_answer) == 
                                       self.normalize_answer(s.ground_truth_answer))
                        accuracy = correct / len(all_evaluation_samples)
                        pbar.set_description(f"Evaluating A-OKVQA (Acc: {accuracy:.3f})")
                
                # Print some examples
                if batch_idx < 3 and self.rank == 0:
                    for i, sample in enumerate(batch_samples[:2]):  # Show first 2 samples
                        print(f"\nExample {batch_idx}-{i}:")
                        print(f"  Question: {sample.question[:100]}...")
                        print(f"  Ground Truth: {sample.ground_truth_answer}")
                        print(f"  Predicted: {sample.predicted_answer}")
                        print(f"  Generated: {sample.metadata['generated_text'][:200]}...")
                
            except Exception as e:
                if self.rank == 0:
                    print(f"Error evaluating batch {batch_idx}: {e}")
                continue
        
        if self.rank == 0:
            pbar.close()
        
        # Synchronize results across processes if distributed
        if self.world_size > 1:
            # Gather results from all processes
            gathered_samples = [None] * self.world_size
            dist.all_gather_object(gathered_samples, all_evaluation_samples)
            
            # Flatten results
            if self.rank == 0:
                all_evaluation_samples = []
                for samples in gathered_samples:
                    all_evaluation_samples.extend(samples)
        
        # Calculate metrics
        if self.rank == 0:
            metrics = calculate_vqa_accuracy(all_evaluation_samples, self.normalize_answer)
            
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            print(f"Total samples: {metrics.total_samples}")
            print(f"Exact match accuracy: {metrics.exact_match_accuracy:.4f}")
            print(f"Normalized accuracy: {metrics.normalized_accuracy:.4f}")
            if metrics.multiple_choice_accuracy is not None:
                print(f"Multiple choice accuracy: {metrics.multiple_choice_accuracy:.4f}")
            print("=" * 60)
            
            # Save detailed results if requested
            if save_results:
                if results_path is None:
                    results_path = f"evaluation_results_stage_{scheduled_stage}.json"
                
                self.save_evaluation_results(all_evaluation_samples, metrics, results_path)
                print(f"Detailed results saved to: {results_path}")
            
            return metrics
        else:
            # Non-main processes return empty metrics
            return VQAAccuracyMetrics(0, 0.0, 0.0, None, None)
    
    def save_evaluation_results(self,
                               evaluation_samples: List[EvaluationSample],
                               metrics: VQAAccuracyMetrics,
                               results_path: str):
        """
        Save detailed evaluation results to JSON file
        
        Args:
            evaluation_samples: List of evaluation samples
            metrics: Calculated metrics
            results_path: Path to save results
        """
        results = {
            'metrics': {
                'total_samples': metrics.total_samples,
                'exact_match_accuracy': metrics.exact_match_accuracy,
                'normalized_accuracy': metrics.normalized_accuracy,
                'multiple_choice_accuracy': metrics.multiple_choice_accuracy,
                'correct_samples': metrics.correct_samples
            },
            'samples': []
        }
        
        # Add sample details
        for sample in evaluation_samples:
            sample_dict = {
                'question_id': sample.question_id,
                'question': sample.question,
                'ground_truth_answer': sample.ground_truth_answer,
                'predicted_answer': sample.predicted_answer,
                'correct': (self.normalize_answer(sample.predicted_answer) == 
                           self.normalize_answer(sample.ground_truth_answer)),
                'choices': sample.choices,
                'generated_text': sample.metadata.get('generated_text', '') if sample.metadata else ''
            }
            results['samples'].append(sample_dict)
        
        # Save to file
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)