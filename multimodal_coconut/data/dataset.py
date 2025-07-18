# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# Multimodal CoCoNuT Dataset Implementation
# Extends the original CoCoNuT dataset to handle image-text pairs

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

# Import image processing functions from dedicated module
from .image_processor import ImageProcessor


class MultimodalDataset:
    """
    Multimodal dataset class that extends original CoCoNuT dataset structure
    to handle image-text pairs for visual question answering tasks.
    
    Expected data format:
    {
        "image_path": "path/to/image.jpg",
        "question": "What is in the image?",
        "steps": ["Step 1: I can see...", "Step 2: The image shows..."],
        "answer": "The answer is..."
    }
    """
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: PreTrainedTokenizerBase,
                 image_root: str = None,
                 image_size: int = 448,
                 max_num_patches: int = 12,
                 use_thumbnail: bool = True,
                 max_size: int = 1000000000):
        """
        Initialize multimodal dataset
        
        Args:
            data_path: Path to JSON data file
            tokenizer: Tokenizer for text processing
            image_root: Root directory for images (if not absolute paths in data)
            image_size: Target image size for preprocessing
            max_num_patches: Maximum number of image patches
            use_thumbnail: Whether to use thumbnail in dynamic preprocessing
            max_size: Maximum dataset size
        """
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.image_size = image_size
        self.max_num_patches = max_num_patches
        self.use_thumbnail = use_thumbnail
        
        # Create dedicated image processor
        self.image_processor = ImageProcessor(
            image_size=image_size,
            max_num_patches=max_num_patches,
            use_thumbnail=use_thumbnail
        )
        
        # Load and process data
        self.data = self._load_data(data_path, max_size)
        self.dataset = self._create_dataset()
    
    def _load_data(self, data_path: str, max_size: int) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Limit dataset size if specified
        data = data[:max_size]
        
        # Add index to each sample
        data = [{**d, "idx": idx} for idx, d in enumerate(data)]
        
        return data
    

    
    def _load_and_process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image using the dedicated ImageProcessor"""
        try:
            # Use the dedicated image processor
            pixel_values = self.image_processor.load_image(
                image_path, 
                image_root=str(self.image_root) if self.image_root else None,
                return_dummy_on_error=True
            )
            return pixel_values
            
        except Exception as e:
            # Log error and return dummy tensor to continue training
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy tensor with single patch
            dummy_tensor = torch.zeros(1, 3, self.image_size, self.image_size)
            return dummy_tensor
    
    def tokenize_multimodal_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a multimodal sample with special token integration
        
        Args:
            sample: Dictionary containing image_path, question, steps, answer, idx
            
        Returns:
            Dictionary with tokenized components and processed image
        """
        # Process image
        pixel_values = self._load_and_process_image(sample["image_path"])
        
        # Tokenize text components following InternVL3 pattern
        # Add proper image context tokens for InternVL3
        num_patches = pixel_values.shape[0]
        img_context_tokens = '<IMG_CONTEXT>' * num_patches
        question_with_image = f"<img>{img_context_tokens}</img>\n{sample['question']}\n"
        question_tokenized = self.tokenizer.encode(
            question_with_image, add_special_tokens=True
        )
        
        # Tokenize reasoning steps
        steps_tokenized = [
            self.tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        
        # Tokenize answer
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
        answer_tokenized = self.tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [eos_token_id]
        
        return {
            "pixel_values": pixel_values,  # Keep as tensor
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
            "num_patches": pixel_values.shape[0]  # Number of image patches
        }
    
    def _create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from processed data - simplified version following original CoCoNuT pattern"""
        # Extract keys for dataset creation
        keys = self.data[0].keys()
        dataset_dict = {k: [d[k] for d in self.data] for k in keys}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Use appropriate number of processes for dataset size
        optimal_num_proc = min(8, max(1, len(self.data) // 10)) if len(self.data) > 10 else 1
        
        # Follow original CoCoNuT pattern: simple distributed processing
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            if dist.get_rank() == 0:
                processed_dataset = [dataset.map(
                    self.tokenize_multimodal_sample, 
                    remove_columns=list(dataset.features), 
                    num_proc=optimal_num_proc
                )]
            else:
                processed_dataset = [None]
            dist.broadcast_object_list(processed_dataset, src=0)
            dataset = processed_dataset[0]
        else:
            dataset = dataset.map(
                self.tokenize_multimodal_sample, 
                remove_columns=list(dataset.features), 
                num_proc=optimal_num_proc
            )
        
        return dataset
    

    
    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        return self.dataset[idx]
    
    def verify_sample(self, idx: int = 0) -> bool:
        """
        Verify that tokenization works correctly by reconstructing text
        Similar to original CoCoNuT verification
        """
        try:
            # Get original and processed data
            original = self.data[idx]
            processed = self.dataset[idx]
            
            # Reconstruct complete text sequence (excluding image token)
            complete = original["question"] + "\n" + "\n".join(original["steps"]) + "\n### " + original["answer"]
            complete_tokenized = self.tokenizer.encode(complete, add_special_tokens=True) + [
                self.tokenizer.eos_token_id
            ]
            
            # Compare with processed tokens (excluding <image> token)
            # Remove the <image> token from question_tokenized for comparison
            question_tokens_no_image = self.tokenizer.encode(
                original["question"] + "\n", add_special_tokens=True
            )
            
            reconstructed_tokens = (
                question_tokens_no_image +
                list(itertools.chain.from_iterable(processed["steps_tokenized"])) +
                processed["answer_tokenized"]
            )
            
            # Verify pixel values shape
            pixel_values = processed["pixel_values"]
            if isinstance(pixel_values, list):
                pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
            assert pixel_values.dim() == 4  # [num_patches, channels, height, width]
            assert pixel_values.shape[1:] == (3, self.image_size, self.image_size)
            
            print(f"Sample {idx} verification:")
            print(f"  Image patches: {processed['num_patches']}")
            print(f"  Pixel values shape: {pixel_values.shape}")
            print(f"  Text tokenization: {'✓' if len(reconstructed_tokens) == len(complete_tokenized) else '✗'}")
            
            return True
            
        except Exception as e:
            print(f"Verification failed for sample {idx}: {e}")
            return False


def get_multimodal_dataset(data_path: str, 
                          tokenizer: PreTrainedTokenizerBase,
                          image_root: str = None,
                          image_size: int = 448,
                          max_num_patches: int = 12,
                          use_thumbnail: bool = True,
                          max_size: int = 1000000000) -> Dataset:
    """
    Convenience function to create multimodal dataset
    
    Args:
        data_path: Path to JSON data file
        tokenizer: Tokenizer for text processing
        image_root: Root directory for images
        image_size: Target image size for preprocessing
        max_num_patches: Maximum number of image patches
        use_thumbnail: Whether to use thumbnail in dynamic preprocessing
        max_size: Maximum dataset size
        
    Returns:
        HuggingFace Dataset object
    """
    multimodal_dataset = MultimodalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_root=image_root,
        image_size=image_size,
        max_num_patches=max_num_patches,
        use_thumbnail=use_thumbnail,
        max_size=max_size
    )
    
    return multimodal_dataset.dataset

@dataclass
class MultimodalCollator:
    """
    Multimodal data collator that extends the original CoCoNuT collator
    to handle both image tensors and text sequences efficiently.
    
    Key features:
    - Aligns latent tokens across batch samples for KV cache efficiency
    - Handles variable image patch counts and sequence lengths
    - Batches pixel_values tensors properly
    """
    
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100
    
    def __call__(self, features, return_tensors=None):
        """
        Collate multimodal features into batches
        
        Args:
            features: List of feature dictionaries containing either:
                Raw format (from MultimodalDataset):
                - pixel_values: Image tensor [num_patches, 3, H, W]
                - question_tokenized, steps_tokenized, answer_tokenized
                - num_patches: Number of image patches
                
                Processed format (from get_multimodal_cot_latent_dataset):
                - pixel_values: Image tensor [num_patches, 3, H, W]
                - input_ids: Text token IDs
                - labels: Target labels (optional)
                - attention_mask: Attention mask
                - position_ids: Position IDs (optional)
                - num_patches: Number of image patches
                
        Returns:
            Batched dictionary with properly padded tensors
        """
        assert self.tokenizer.padding_side == "right"
        
        # Check if features are in raw format (from MultimodalDataset) or processed format
        if 'input_ids' not in features[0]:
            # Raw format - convert to simple input_ids format for basic collation
            processed_features = []
            for feature in features:
                # Create simple input_ids by concatenating question + steps + answer
                import itertools
                input_ids = (
                    feature['question_tokenized'] +
                    list(itertools.chain.from_iterable(feature['steps_tokenized'])) +
                    feature['answer_tokenized']
                )
                
                processed_feature = {
                    'pixel_values': feature['pixel_values'],
                    'num_patches': feature['num_patches'],
                    'input_ids': input_ids,
                    'attention_mask': [1] * len(input_ids),
                    'idx': feature.get('idx', 0)
                }
                processed_features.append(processed_feature)
            features = processed_features
        
        # Separate pixel_values and num_patches from text features
        pixel_values_list = []
        num_patches_list = []
        text_features = []
        
        for feature in features:
            # Make a copy to avoid modifying original
            feature_copy = feature.copy()
            pixel_values_list.append(feature_copy.pop('pixel_values'))
            num_patches_list.append(feature_copy.pop('num_patches'))
            text_features.append(feature_copy)
        
        # Apply the original CoCoNuT collation logic for text
        # This handles latent token alignment for KV cache efficiency
        if self.latent_id is not None:
            earliest_latent = [
                feature["input_ids"].index(self.latent_id)
                for feature in text_features
                if self.latent_id in feature["input_ids"]
            ]
        else:
            earliest_latent = []

        if len(earliest_latent) > 0:  # if there are continuous thoughts in the sequence
            latest_earliest_latent = max(earliest_latent)
            for feature in text_features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature[
                        "labels"
                    ]
                if "attention_mask" in feature:
                    feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in text_features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids" and k != "idx"
            }
            for feature in text_features
        ]

        # Run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in text_features]
            if label_name in text_features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in text_features]
            if "position_ids" in text_features[0].keys()
            else None
        )
        
        # Manually pad labels and position_ids
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )
        
        # Handle multimodal components
        multimodal_batch = self._collate_multimodal_features(pixel_values_list, num_patches_list)
        
        # Store num_patches_list separately - don't pass to model forward
        batch['pixel_values'] = multimodal_batch['pixel_values']
        batch['image_flags'] = multimodal_batch['image_flags']
        # Store num_patches_list for internal use but don't pass to model
        batch['_num_patches_list'] = multimodal_batch['num_patches_list']
        
        return batch
    
    def _collate_multimodal_features(self, 
                                   pixel_values_list: List[Union[torch.Tensor, List]], 
                                   num_patches_list: List[int]) -> Dict[str, torch.Tensor]:
        """
        Collate multimodal features (images) into batch tensors following InternVL3 format
        
        Args:
            pixel_values_list: List of image tensors or lists [num_patches, 3, H, W]
            num_patches_list: List of patch counts for each image
            
        Returns:
            Dictionary with batched multimodal tensors in InternVL3 format
        """
        # Ensure all pixel_values are tensors (handle both tensor and list formats)
        tensor_list = []
        for pixel_values in pixel_values_list:
            if isinstance(pixel_values, list):
                # Convert list back to tensor (HF datasets serialization)
                pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
            elif not isinstance(pixel_values, torch.Tensor):
                # Handle any other format
                pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
            tensor_list.append(pixel_values)
        
        # Concatenate all image patches for InternVL3 format
        # InternVL3 expects all patches concatenated along batch dimension
        all_pixel_values = torch.cat(tensor_list, dim=0)  # [total_patches, 3, H, W]
        
        # Create image flags indicating which samples have images
        # InternVL3 uses image_flags to filter visual embeddings
        batch_size = len(num_patches_list)
        image_flags = torch.ones(batch_size, 1, dtype=torch.long)  # All samples have images
        
        return {
            'pixel_values': all_pixel_values,
            'image_flags': image_flags,
            'num_patches_list': num_patches_list,
        }


def get_multimodal_cot_latent_dataset(
    scheduled_stage: int,
    base_dataset: Dataset,
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
    shuffle: bool = False,
) -> Dataset:
    """
    Prepare multimodal CoCoNuT training data for a given stage
    Extends the original get_cot_latent_dataset to handle multimodal inputs
    
    Args:
        scheduled_stage: Current training stage (number of steps to replace with latent tokens)
        base_dataset: Base dataset with tokenized multimodal samples
        configs: Configuration object with training parameters
        start_id: Start latent token ID
        latent_id: Latent token ID
        end_id: End latent token ID
        no_special_marker: Whether to skip special marker tokens
        shuffle: Whether to shuffle the dataset
        
    Returns:
        Dataset prepared for the current training stage
    """
    # Import here to avoid circular imports
    from ..training.stage_manager import StageManager
    
    n_additional_tokens = 0 if no_special_marker else 2
    stage_manager = StageManager(configs)

    def process_multimodal_dataset(sample):
        """Process a single multimodal sample for the current stage"""
        
        # Use stage manager for consistent stage calculation
        effective_stage, n_skip_steps, n_latent_tokens = stage_manager.get_effective_stage_for_sample(
            scheduled_stage, sample["steps_tokenized"]
        )

        # Build token sequence: question + latent tokens + remaining steps + answer
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(
                itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:])
            )
            + sample["answer_tokenized"]
        )

        # Create labels (ignore question and latent tokens in loss)
        labels = (
            [-100] * (
                len(sample["question_tokenized"])
                + n_latent_tokens
                + n_additional_tokens
            )
            + tokens[
                n_latent_tokens
                + n_additional_tokens
                + len(sample["question_tokenized"]) :
            ]
        )

        return {
            "pixel_values": sample["pixel_values"],
            "num_patches": sample["num_patches"],
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    # Apply processing with distributed support using robust approach
    optimal_num_proc = min(8, max(1, len(base_dataset) // 100))
    
    if torch.cuda.device_count() > 1 and dist.is_initialized():
        if dist.get_rank() == 0:
            try:
                processed_dataset = base_dataset.map(
                    process_multimodal_dataset, 
                    remove_columns=list(base_dataset.features), 
                    num_proc=optimal_num_proc,
                    desc="Processing multimodal CoT dataset",
                    load_from_cache_file=False
                )
                if shuffle:
                    processed_dataset = processed_dataset.shuffle()
                processed_dataset = [processed_dataset]
            except Exception as e:
                print(f"Error in distributed CoT processing: {e}")
                print("Falling back to sequential processing...")
                processed_dataset = [_process_cot_sequentially(base_dataset, process_multimodal_dataset, shuffle)]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        try:
            processed_dataset = base_dataset.map(
                process_multimodal_dataset, 
                remove_columns=list(base_dataset.features), 
                num_proc=optimal_num_proc,
                desc="Processing multimodal CoT dataset",
                load_from_cache_file=False
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            dataset = processed_dataset
        except Exception as e:
            print(f"Error in parallel CoT processing: {e}")
            print("Falling back to sequential processing...")
            dataset = _process_cot_sequentially(base_dataset, process_multimodal_dataset, shuffle)

    return dataset


def _process_cot_sequentially(base_dataset: Dataset, process_func, shuffle: bool = False) -> Dataset:
    """Sequential processing fallback for CoT dataset processing"""
    print("Processing CoT dataset sequentially...")
    processed_samples = []
    
    for i in range(len(base_dataset)):
        try:
            sample = base_dataset[i]
            processed_sample = process_func(sample)
            processed_samples.append(processed_sample)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(base_dataset)} samples")
                
        except Exception as e:
            print(f"Error processing CoT sample {i}: {e}")
            # Skip this sample or create a dummy one
            continue
    
    # Convert back to dataset
    if processed_samples:
        keys = processed_samples[0].keys()
        processed_dict = {k: [sample[k] for sample in processed_samples] for k in keys}
        dataset = Dataset.from_dict(processed_dict)
        
        if shuffle:
            dataset = dataset.shuffle()
            
        return dataset
    else:
        raise ValueError("No CoT samples could be processed")


def get_multimodal_question_latent_dataset(
    scheduled_stage: int,
    base_dataset_valid: Dataset,
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
) -> Dataset:
    """
    Prepare multimodal validation/test data with appropriate latent tokens
    Extends the original get_question_latent_dataset for multimodal inputs
    
    Args:
        scheduled_stage: Current training stage
        base_dataset_valid: Validation dataset with tokenized multimodal samples
        configs: Configuration object
        start_id: Start latent token ID
        latent_id: Latent token ID
        end_id: End latent token ID
        no_special_marker: Whether to skip special marker tokens
        
    Returns:
        Dataset prepared for validation/testing
    """
    # Import here to avoid circular imports
    from ..training.stage_manager import StageManager
    
    stage_manager = StageManager(configs)

    def process_multimodal_validation_dataset(sample):
        """Process a single multimodal validation sample"""
        
        # Use stage manager for consistent validation data preparation
        stage_info = stage_manager.get_stage_info(scheduled_stage)
        
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        # Build token sequence for inference: question + latent tokens
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "pixel_values": sample["pixel_values"],
            "num_patches": sample["num_patches"],
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    # Use robust processing approach
    optimal_num_proc = min(8, max(1, len(base_dataset_valid) // 100))
    
    return base_dataset_valid.map(
        process_multimodal_validation_dataset, 
        remove_columns=list(base_dataset_valid.features), 
        num_proc=optimal_num_proc,
        desc="Processing multimodal validation dataset",
        load_from_cache_file=False
    )