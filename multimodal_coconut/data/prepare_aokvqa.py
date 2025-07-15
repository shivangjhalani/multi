#!/usr/bin/env python3
"""
A-OKVQA Dataset Preparation for Multimodal CoCoNuT

This script downloads and preprocesses the A-OKVQA dataset from HuggingFace,
converting it to the format expected by multimodal CoCoNuT training.

The conversion process:
1. Downloads A-OKVQA dataset from HuggingFace (HuggingFaceM4/A-OKVQA)
2. Saves images locally with proper organization
3. Converts the format to multimodal CoCoNuT structure:
   - image_path: path to saved image
   - question: original question + choices formatted for reasoning
   - steps: rationales as reasoning steps
   - answer: correct choice or direct answer

Usage:
    python prepare_aokvqa.py --output_dir data/aokvqa --splits train val test
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

from PIL import Image


def build_question_with_choices(question: str, choices: List[str]) -> str:
    """
    Build question text with multiple choice options
    
    Args:
        question: Original question text
        choices: List of choice options
        
    Returns:
        Formatted question with choices
    """
    if not choices:
        return question
    
    choice_str = ', '.join([f'{idx}: {choice}' for idx, choice in enumerate(choices)])
    return f'{question} The choices are {choice_str}.'


def format_rationales_as_steps(rationales: List[str]) -> List[str]:
    """
    Format rationales as reasoning steps for CoCoNuT training
    
    Args:
        rationales: List of rationale strings
        
    Returns:
        List of formatted reasoning steps
    """
    if not rationales:
        return ["Let me analyze this question step by step."]
    
    # Format each rationale as a reasoning step
    steps = []
    for i, rationale in enumerate(rationales):
        if len(rationales) == 1:
            steps.append(f"Let me think about this: {rationale}")
        else:
            steps.append(f"Step {i+1}: {rationale}")
    
    return steps


def get_answer_text(example: Dict[str, Any], use_direct_answer: bool = False) -> str:
    """
    Extract answer text from A-OKVQA example
    
    Args:
        example: A-OKVQA example dictionary
        use_direct_answer: Whether to use direct answer instead of choice
        
    Returns:
        Answer text
    """
    if use_direct_answer and 'direct_answers' in example and example['direct_answers']:
        # Use the most common direct answer
        direct_answers = example['direct_answers']
        if isinstance(direct_answers, list):
            # Take the first answer or most common one
            return direct_answers[0] if direct_answers else "unknown"
        else:
            return str(direct_answers)
    
    # Use multiple choice answer
    choices = example.get('choices', [])
    correct_idx = example.get('correct_choice_idx')
    
    if choices and correct_idx is not None and 0 <= correct_idx < len(choices):
        return choices[correct_idx]
    
    return "unknown"


def process_split(split_name: str, 
                 output_dir: Path, 
                 use_direct_answer: bool = False,
                 max_samples: Optional[int] = None) -> None:
    """
    Process a single split of the A-OKVQA dataset
    
    Args:
        split_name: Name of the split ('train', 'validation', 'test')
        output_dir: Output directory for processed data
        use_direct_answer: Whether to use direct answers instead of choices
        max_samples: Maximum number of samples to process (for testing)
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library is required. Install with: pip install datasets")
    
    print(f"Loading A-OKVQA {split_name} split...")
    
    # Load dataset from HuggingFace
    try:
        ds = load_dataset('HuggingFaceM4/A-OKVQA', split=split_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have internet connection and datasets library installed")
        return
    
    # Create directories
    images_dir = output_dir / 'images' / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    
    output_records = []
    
    # Process samples
    samples_to_process = min(len(ds), max_samples) if max_samples else len(ds)
    print(f"Processing {samples_to_process} samples from {split_name} split...")
    
    for idx, example in enumerate(tqdm(ds, desc=f'Processing {split_name}')):
        if max_samples and idx >= max_samples:
            break
            
        try:
            # Extract components
            image = example['image']
            question_id = example['question_id']
            question_text = example.get('question', '')
            choices = example.get('choices', [])
            rationales = example.get('rationales', [])
            
            # Skip if essential components are missing
            if not question_text or not image:
                print(f"Skipping sample {idx}: missing question or image")
                continue
            
            # For test split, we might not have rationales or correct answers
            if split_name == 'test':
                if not rationales:
                    rationales = ["I need to analyze this image and question carefully."]
            else:
                # For train/val, skip if no rationales
                if not rationales:
                    print(f"Skipping sample {idx}: no rationales available")
                    continue
            
            # Save image
            img_name = f'{question_id}.jpg'
            img_path = images_dir / img_name
            
            if not img_path.exists():
                try:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(img_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"Error saving image {img_name}: {e}")
                    continue
            
            # Build question with choices (for multiple choice reasoning)
            if choices:
                formatted_question = build_question_with_choices(question_text, choices)
            else:
                formatted_question = question_text
            
            # Format rationales as reasoning steps
            reasoning_steps = format_rationales_as_steps(rationales)
            
            # Get answer
            answer_text = get_answer_text(example, use_direct_answer)
            
            # Create record in multimodal CoCoNuT format
            record = {
                'image_path': str(img_path.relative_to(output_dir)),  # Relative path
                'question': formatted_question,
                'steps': reasoning_steps,
                'answer': answer_text,
                'metadata': {
                    'question_id': question_id,
                    'original_question': question_text,
                    'choices': choices,
                    'split': split_name,
                    'use_direct_answer': use_direct_answer
                }
            }
            
            output_records.append(record)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save processed data
    output_file = output_dir / f'{split_name}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(output_records)} samples to {output_file}")
    
    # Print statistics
    if output_records:
        avg_steps = sum(len(r['steps']) for r in output_records) / len(output_records)
        avg_question_len = sum(len(r['question'].split()) for r in output_records) / len(output_records)
        print(f"Statistics for {split_name}:")
        print(f"  - Average reasoning steps: {avg_steps:.1f}")
        print(f"  - Average question length: {avg_question_len:.1f} words")


def verify_dataset(output_dir: Path, split_name: str = 'train') -> bool:
    """
    Verify the processed dataset by checking a few samples
    
    Args:
        output_dir: Directory containing processed data
        split_name: Split to verify
        
    Returns:
        True if verification passes
    """
    try:
        # Load processed data
        data_file = output_dir / f'{split_name}.json'
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            return False
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("No data found in file")
            return False
        
        print(f"Verification for {split_name} split:")
        print(f"  - Total samples: {len(data)}")
        
        # Check first sample
        sample = data[0]
        required_fields = ['image_path', 'question', 'steps', 'answer']
        
        for field in required_fields:
            if field not in sample:
                print(f"  ✗ Missing required field: {field}")
                return False
        
        # Check image exists
        image_path = output_dir / sample['image_path']
        if not image_path.exists():
            print(f"  ✗ Image file not found: {image_path}")
            return False
        
        # Try to load image
        try:
            img = Image.open(image_path)
            print(f"  ✓ Image loaded successfully: {img.size}")
        except Exception as e:
            print(f"  ✗ Error loading image: {e}")
            return False
        
        print(f"  ✓ Sample structure valid")
        print(f"  ✓ Question: {sample['question'][:100]}...")
        print(f"  ✓ Steps: {len(sample['steps'])} reasoning steps")
        print(f"  ✓ Answer: {sample['answer']}")
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Prepare A-OKVQA dataset for multimodal CoCoNuT')
    parser.add_argument('--output_dir', type=str, default='data/aokvqa',
                       help='Output directory for processed dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Dataset splits to process')
    parser.add_argument('--use_direct_answer', action='store_true',
                       help='Use direct answers instead of multiple choice')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing dataset without processing')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verify_only:
        # Only verify existing data
        for split in args.splits:
            print(f"\nVerifying {split} split...")
            verify_dataset(output_dir, split)
    else:
        # Process dataset
        print("Starting A-OKVQA dataset preparation...")
        print(f"Output directory: {output_dir}")
        print(f"Splits to process: {args.splits}")
        print(f"Use direct answers: {args.use_direct_answer}")
        
        if args.max_samples:
            print(f"Max samples per split: {args.max_samples}")
        
        # Process each split
        for split in args.splits:
            print(f"\n{'='*50}")
            print(f"Processing {split} split")
            print(f"{'='*50}")
            
            try:
                process_split(
                    split_name=split,
                    output_dir=output_dir,
                    use_direct_answer=args.use_direct_answer,
                    max_samples=args.max_samples
                )
            except Exception as e:
                print(f"Error processing {split}: {e}")
                continue
        
        # Verify processed data
        print(f"\n{'='*50}")
        print("Verification")
        print(f"{'='*50}")
        
        for split in args.splits:
            if split == 'validation':
                # HuggingFace uses 'validation', but we save as 'val'
                verify_split = 'validation'
            else:
                verify_split = split
            print(f"\nVerifying {verify_split}...")
            verify_dataset(output_dir, verify_split)
    
    print("\nA-OKVQA dataset preparation completed!")


if __name__ == '__main__':
    main()