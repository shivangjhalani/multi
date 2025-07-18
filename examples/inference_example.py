#!/usr/bin/env python3
"""
Inference Example for Multimodal CoCoNuT

This script demonstrates how to use a trained Multimodal CoCoNuT model for inference
on image-question pairs. It shows both single sample and batch inference.

Usage:
    python examples/inference_example.py --model path/to/checkpoint --image path/to/image.jpg --question "What is in the image?"
    python examples/inference_example.py --model path/to/checkpoint --batch-file examples/sample_questions.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
from transformers import AutoTokenizer
from multimodal_coconut import MultimodalCoconut, load_config
from multimodal_coconut.data import ImageProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multimodal CoCoNuT Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint or model ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question about the image"
    )
    parser.add_argument(
        "--batch-file",
        type=str,
        default=None,
        help="JSON file with batch of image-question pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of greedy decoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up compute device"""
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def load_model_and_tokenizer(model_path: str, config_path: str = None, device: torch.device = None):
    """
    Load trained model and tokenizer
    
    Args:
        model_path: Path to model checkpoint or model ID
        config_path: Optional path to configuration file
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer, image_processor)
    """
    print(f"Loading model from: {model_path}")
    
    # Load configuration
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    else:
        # Use default configuration
        from multimodal_coconut import create_config_from_template
        config = create_config_from_template('default')
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
    except:
        # Fallback to base model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            use_fast=False
        )
    
    # Ensure special tokens are present
    special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    try:
        model = MultimodalCoconut.from_pretrained(model_path)
    except:
        print("Warning: Could not load checkpoint, creating new model")
        # Create new model (for demonstration)
        from transformers import AutoModel
        base_model = AutoModel.from_pretrained(
            config.model_id,
            trust_remote_code=True
        )
        
        # Get special token IDs
        latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        eos_token_id = tokenizer.eos_token_id
        
        model = MultimodalCoconut(
            base_model=base_model,
            latent_token_id=latent_token_id,
            start_latent_id=start_latent_id,
            end_latent_id=end_latent_id,
            eos_token_id=eos_token_id
        )
    
    # Move to device
    if device is not None:
        model = model.to(device)
    
    model.eval()
    
    # Create image processor
    image_processor = ImageProcessor(
        image_size=config.get('image_size', 448),
        max_num_patches=config.get('max_num_patches', 12),
        use_thumbnail=config.get('use_thumbnail', True),
        dynamic_preprocess=config.get('dynamic_preprocess', True)
    )
    
    print("Model loaded successfully")
    return model, tokenizer, image_processor


def prepare_inputs(image_path: str, question: str, tokenizer, image_processor, device: torch.device):
    """
    Prepare inputs for inference
    
    Args:
        image_path: Path to image file
        question: Question text
        tokenizer: Tokenizer instance
        image_processor: Image processor instance
        device: Device to put tensors on
        
    Returns:
        Dictionary with prepared inputs
    """
    # Load and process image
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = image_processor.process_image(image_path)
        pixel_values = pixel_values.unsqueeze(0).to(device)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        pixel_values = None
    
    # Tokenize question
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def generate_response(model, inputs: Dict[str, torch.Tensor], tokenizer, generation_config: Dict[str, Any]):
    """
    Generate response using the model
    
    Args:
        model: MultimodalCoconut model
        inputs: Prepared inputs dictionary
        tokenizer: Tokenizer instance
        generation_config: Generation configuration
        
    Returns:
        Generated response text
    """
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            **generation_config
        )
    
    # Decode response (skip input tokens)
    input_length = inputs["input_ids"].shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response.strip()


def single_inference(args, model, tokenizer, image_processor, device):
    """Run inference on a single image-question pair"""
    print(f"\nProcessing single inference:")
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    
    # Prepare inputs
    inputs = prepare_inputs(args.image, args.question, tokenizer, image_processor, device)
    
    # Generation configuration
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    # Generate response
    print("Generating response...")
    response = generate_response(model, inputs, tokenizer, generation_config)
    
    print(f"\nResponse: {response}")
    
    return {
        "image": args.image,
        "question": args.question,
        "answer": response
    }


def batch_inference(args, model, tokenizer, image_processor, device):
    """Run inference on a batch of image-question pairs"""
    print(f"\nProcessing batch inference from: {args.batch_file}")
    
    # Load batch data
    with open(args.batch_file, 'r') as f:
        batch_data = json.load(f)
    
    if not isinstance(batch_data, list):
        raise ValueError("Batch file should contain a list of samples")
    
    results = []
    
    # Generation configuration
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    for i, sample in enumerate(batch_data):
        print(f"\nProcessing sample {i+1}/{len(batch_data)}")
        
        if "image" not in sample or "question" not in sample:
            print(f"Skipping sample {i+1}: missing 'image' or 'question' field")
            continue
        
        image_path = sample["image"]
        question = sample["question"]
        
        print(f"Image: {image_path}")
        print(f"Question: {question}")
        
        try:
            # Prepare inputs
            inputs = prepare_inputs(image_path, question, tokenizer, image_processor, device)
            
            # Generate response
            response = generate_response(model, inputs, tokenizer, generation_config)
            
            print(f"Answer: {response}")
            
            result = {
                "image": image_path,
                "question": question,
                "answer": response
            }
            
            # Include ground truth if available
            if "answer" in sample:
                result["ground_truth"] = sample["answer"]
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            results.append({
                "image": image_path,
                "question": question,
                "answer": f"Error: {str(e)}",
                "error": True
            })
    
    return results


def main():
    """Main inference function"""
    args = parse_args()
    
    # Validate arguments
    if not args.batch_file and (not args.image or not args.question):
        print("Error: Either provide --image and --question, or --batch-file")
        sys.exit(1)
    
    # Set up device
    device = setup_device(args.device)
    
    # Load model and tokenizer
    model, tokenizer, image_processor = load_model_and_tokenizer(
        args.model, args.config, device
    )
    
    # Run inference
    if args.batch_file:
        results = batch_inference(args, model, tokenizer, image_processor, device)
    else:
        result = single_inference(args, model, tokenizer, image_processor, device)
        results = [result]
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print(f"\nInference completed successfully!")
    print(f"Processed {len(results)} samples")


if __name__ == "__main__":
    main()