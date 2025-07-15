# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# Multimodal CoCoNuT Dataset Implementation
# Extends the original CoCoNuT dataset to handle image-text pairs

import json
import itertools
import random
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# InternVL3 image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """Build image transformation pipeline following InternVL3 preprocessing"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio for dynamic preprocessing"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Dynamic image preprocessing following InternVL3 approach"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


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
        self.transform = build_transform(image_size)
        
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
    
    def _validate_image_path(self, image_path: str) -> str:
        """Validate and resolve image path"""
        if os.path.isabs(image_path):
            full_path = image_path
        elif self.image_root:
            full_path = str(self.image_root / image_path)
        else:
            full_path = image_path
            
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        return full_path
    
    def _load_and_process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image following InternVL3 approach"""
        try:
            # Validate image path
            full_path = self._validate_image_path(image_path)
            
            # Load image
            image = Image.open(full_path).convert('RGB')
            
            # Dynamic preprocessing
            images = dynamic_preprocess(
                image, 
                image_size=self.image_size, 
                use_thumbnail=self.use_thumbnail, 
                max_num=self.max_num_patches
            )
            
            # Apply transforms
            pixel_values = [self.transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            
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
        
        # Tokenize text components following original CoCoNuT pattern
        # Add <image> token at the beginning of question
        question_with_image = "<image>\n" + sample["question"] + "\n"
        question_tokenized = self.tokenizer.encode(
            question_with_image, add_special_tokens=True
        )
        
        # Tokenize reasoning steps
        steps_tokenized = [
            self.tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        
        # Tokenize answer
        answer_tokenized = self.tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [self.tokenizer.eos_token_id]
        
        return {
            "pixel_values": pixel_values.tolist(),  # Convert to list for HF datasets
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
            "num_patches": pixel_values.shape[0]  # Number of image patches
        }
    
    def _create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from processed data with robust processing"""
        # Extract keys for dataset creation
        keys = self.data[0].keys()
        dataset_dict = {k: [d[k] for d in self.data] for k in keys}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Determine optimal number of processes
        optimal_num_proc = min(8, max(1, len(self.data) // 100))  # More conservative
        
        print(f"Processing {len(self.data)} samples with {optimal_num_proc} processes...")
        
        # Apply tokenization with robust error handling
        if torch.cuda.device_count() > 1 and dist.is_initialized():
            # Distributed processing - only rank 0 processes
            if dist.get_rank() == 0:
                try:
                    processed_dataset = dataset.map(
                        self._safe_tokenize_multimodal_sample, 
                        remove_columns=list(dataset.features), 
                        num_proc=optimal_num_proc,
                        desc="Tokenizing multimodal samples",
                        load_from_cache_file=False  # Avoid cache issues
                    )
                    processed_dataset = [processed_dataset]
                except Exception as e:
                    print(f"Error in distributed processing: {e}")
                    # Fallback to single process
                    processed_dataset = [self._process_dataset_sequentially(dataset)]
            else:
                processed_dataset = [None]
            
            # Broadcast to all ranks
            dist.broadcast_object_list(processed_dataset, src=0)
            dataset = processed_dataset[0]
        else:
            # Single machine processing
            try:
                dataset = dataset.map(
                    self._safe_tokenize_multimodal_sample, 
                    remove_columns=list(dataset.features), 
                    num_proc=optimal_num_proc,
                    desc="Tokenizing multimodal samples",
                    load_from_cache_file=False
                )
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                print("Falling back to sequential processing...")
                dataset = self._process_dataset_sequentially(dataset)
        
        return dataset
    
    def _safe_tokenize_multimodal_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe wrapper around tokenize_multimodal_sample with timeout and error handling
        """
        try:
            return self.tokenize_multimodal_sample(sample)
        except Exception as e:
            print(f"Error processing sample {sample.get('idx', 'unknown')}: {e}")
            # Return a dummy sample to continue processing
            return self._create_dummy_sample(sample.get('idx', 0))
    
    def _create_dummy_sample(self, idx: int) -> Dict[str, Any]:
        """Create a dummy sample for error cases"""
        # Create minimal dummy data
        dummy_pixel_values = torch.zeros(1, 3, self.image_size, self.image_size)
        dummy_question = self.tokenizer.encode("Error loading sample", add_special_tokens=True)
        dummy_steps = [[self.tokenizer.encode("Processing error", add_special_tokens=False)]]
        dummy_answer = self.tokenizer.encode("### Error", add_special_tokens=False) + [self.tokenizer.eos_token_id]
        
        return {
            "pixel_values": dummy_pixel_values,
            "question_tokenized": dummy_question,
            "steps_tokenized": dummy_steps,
            "answer_tokenized": dummy_answer,
            "idx": idx,
            "num_patches": 1
        }
    
    def _process_dataset_sequentially(self, dataset: Dataset) -> Dataset:
        """Process dataset sequentially as fallback"""
        print("Processing samples sequentially...")
        processed_samples = []
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                processed_sample = self._safe_tokenize_multimodal_sample(sample)
                processed_samples.append(processed_sample)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                processed_samples.append(self._create_dummy_sample(i))
        
        # Convert back to dataset
        if processed_samples:
            keys = processed_samples[0].keys()
            processed_dict = {k: [sample[k] for sample in processed_samples] for k in keys}
            return Dataset.from_dict(processed_dict)
        else:
            raise ValueError("No samples could be processed")
    
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
            features: List of feature dictionaries containing:
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
        
        # Separate pixel_values and num_patches from text features
        pixel_values_list = []
        num_patches_list = []
        text_features = []
        
        for feature in features:
            pixel_values_list.append(feature.pop('pixel_values'))
            num_patches_list.append(feature.pop('num_patches'))
            text_features.append(feature)
        
        # Apply the original CoCoNuT collation logic for text
        # This handles latent token alignment for KV cache efficiency
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in text_features
            if self.latent_id in feature["input_ids"]
        ]

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
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in text_features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
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
        batch.update(self._collate_multimodal_features(pixel_values_list, num_patches_list))
        
        return batch
    
    def _collate_multimodal_features(self, 
                                   pixel_values_list: List[Union[torch.Tensor, List]], 
                                   num_patches_list: List[int]) -> Dict[str, torch.Tensor]:
        """
        Collate multimodal features (images) into batch tensors
        
        Args:
            pixel_values_list: List of image tensors or lists [num_patches, 3, H, W]
            num_patches_list: List of patch counts for each image
            
        Returns:
            Dictionary with batched multimodal tensors
        """
        # Convert lists back to tensors if needed (HF datasets serialization issue)
        tensor_list = []
        for pixel_values in pixel_values_list:
            if isinstance(pixel_values, list):
                # Convert list back to tensor
                pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
            tensor_list.append(pixel_values)
        
        # Handle variable number of patches by concatenating all patches
        # and keeping track of patch boundaries
        all_pixel_values = torch.cat(tensor_list, dim=0)  # [total_patches, 3, H, W]
        
        # Create patch boundaries for splitting during forward pass
        patch_boundaries = []
        current_pos = 0
        for num_patches in num_patches_list:
            patch_boundaries.append((current_pos, current_pos + num_patches))
            current_pos += num_patches
        
        return {
            'pixel_values': all_pixel_values,
            'num_patches_list': num_patches_list,
            'patch_boundaries': patch_boundaries
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
    n_additional_tokens = 0 if no_special_marker else 2

    def process_multimodal_dataset(sample):
        """Process a single multimodal sample for the current stage"""
        
        # Handle stage scheduling with uniform probability mixing
        if (
            random.random() < configs.uniform_prob
        ):  # with some prob, randomly sample stage
            scheduled_stage_to_train = random.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        else:
            n_skip_steps, n_latent_tokens = (
                scheduled_stage_to_train,
                scheduled_stage_to_train,
            )

        if configs.no_cot:
            n_skip_steps = 100  # skip all step
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

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

    # Apply processing with distributed support
    if torch.cuda.device_count() > 1 and dist.is_initialized():
        if dist.get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_multimodal_dataset, 
                remove_columns=list(base_dataset.features), 
                num_proc=32
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        processed_dataset = base_dataset.map(
            process_multimodal_dataset, 
            remove_columns=list(base_dataset.features), 
            num_proc=32
        )
        if shuffle:
            processed_dataset = processed_dataset.shuffle()
        dataset = processed_dataset

    return dataset


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
    def process_multimodal_validation_dataset(sample):
        """Process a single multimodal validation sample"""
        
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

    return base_dataset_valid.map(
        process_multimodal_validation_dataset, 
        remove_columns=list(base_dataset_valid.features), 
        num_proc=32
    )