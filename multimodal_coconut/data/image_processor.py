# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Image preprocessing pipeline for Multimodal CoCoNuT
# Integrates InternVL3's dynamic preprocessing with error handling

import os
import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageFile
import numpy as np

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# InternVL3 image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Setup logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processor that integrates InternVL3's dynamic preprocessing
    with robust error handling for corrupted or missing images.
    """
    
    def __init__(self, 
                 image_size: int = 448,
                 max_num_patches: int = 12,
                 min_num_patches: int = 1,
                 use_thumbnail: bool = True,
                 interpolation: InterpolationMode = InterpolationMode.BICUBIC):
        """
        Initialize image processor
        
        Args:
            image_size: Target size for each image patch
            max_num_patches: Maximum number of patches to generate
            min_num_patches: Minimum number of patches to generate
            use_thumbnail: Whether to add thumbnail patch
            interpolation: Interpolation method for resizing
        """
        self.image_size = image_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.use_thumbnail = use_thumbnail
        self.interpolation = interpolation
        
        # Build transformation pipeline
        self.transform = self._build_transform()
        
        # Statistics for monitoring
        self.stats = {
            'processed': 0,
            'errors': 0,
            'corrupted': 0,
            'missing': 0
        }
    
    def _build_transform(self) -> T.Compose:
        """Build image transformation pipeline following InternVL3"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=self.interpolation),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def _find_closest_aspect_ratio(self, 
                                  aspect_ratio: float, 
                                  target_ratios: List[Tuple[int, int]], 
                                  width: int, 
                                  height: int) -> Tuple[int, int]:
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
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        return best_ratio
    
    def dynamic_preprocess(self, image: Image.Image) -> List[Image.Image]:
        """
        Dynamic image preprocessing following InternVL3 approach
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            List of preprocessed image patches
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate possible aspect ratios
        target_ratios = set(
            (i, j) for n in range(self.min_num_patches, self.max_num_patches + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= self.max_num_patches and i * j >= self.min_num_patches
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height
        )

        # Calculate target dimensions
        target_width = self.image_size * target_aspect_ratio[0]
        target_height = self.image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image
        resized_img = image.resize((target_width, target_height), self.interpolation)
        
        # Split into patches
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // self.image_size)) * self.image_size,
                (i // (target_width // self.image_size)) * self.image_size,
                ((i % (target_width // self.image_size)) + 1) * self.image_size,
                ((i // (target_width // self.image_size)) + 1) * self.image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        
        # Add thumbnail if requested and we have multiple patches
        if self.use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.image_size, self.image_size), self.interpolation)
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _create_dummy_image(self) -> torch.Tensor:
        """Create a dummy image tensor for error cases"""
        # Create a simple gradient pattern as dummy image
        dummy_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for i in range(self.image_size):
            dummy_array[i, :, :] = i % 256
        
        dummy_image = Image.fromarray(dummy_array)
        dummy_tensor = self.transform(dummy_image).unsqueeze(0)  # Add batch dimension
        return dummy_tensor
    
    def _validate_image_path(self, image_path: str, image_root: Optional[str] = None) -> str:
        """
        Validate and resolve image path
        
        Args:
            image_path: Path to image file
            image_root: Root directory for relative paths
            
        Returns:
            Absolute path to image file
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        if os.path.isabs(image_path):
            full_path = image_path
        elif image_root:
            full_path = str(Path(image_root) / image_path)
        else:
            full_path = image_path
            
        if not os.path.exists(full_path):
            self.stats['missing'] += 1
            raise FileNotFoundError(f"Image not found: {full_path}")
            
        return full_path
    
    def load_image(self, 
                   image_path: str, 
                   image_root: Optional[str] = None,
                   return_dummy_on_error: bool = True) -> torch.Tensor:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            image_root: Root directory for relative paths
            return_dummy_on_error: Whether to return dummy tensor on error
            
        Returns:
            Preprocessed image tensor of shape [num_patches, 3, image_size, image_size]
        """
        try:
            # Validate image path
            full_path = self._validate_image_path(image_path, image_root)
            
            # Load image
            image = Image.open(full_path).convert('RGB')
            
            # Check if image is valid
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")
            
            # Dynamic preprocessing
            images = self.dynamic_preprocess(image)
            
            # Apply transforms
            pixel_values = [self.transform(img) for img in images]
            pixel_values = torch.stack(pixel_values)
            
            self.stats['processed'] += 1
            return pixel_values
            
        except FileNotFoundError as e:
            logger.warning(f"Image not found: {image_path}")
            if return_dummy_on_error:
                return self._create_dummy_image()
            else:
                raise e
                
        except (OSError, IOError) as e:
            # Handle corrupted images
            logger.warning(f"Corrupted image {image_path}: {e}")
            self.stats['corrupted'] += 1
            if return_dummy_on_error:
                return self._create_dummy_image()
            else:
                raise e
                
        except Exception as e:
            # Handle any other errors
            logger.error(f"Error processing image {image_path}: {e}")
            self.stats['errors'] += 1
            if return_dummy_on_error:
                return self._create_dummy_image()
            else:
                raise e
    
    def load_images_batch(self, 
                         image_paths: List[str], 
                         image_root: Optional[str] = None,
                         return_dummy_on_error: bool = True) -> List[torch.Tensor]:
        """
        Load and preprocess a batch of images
        
        Args:
            image_paths: List of paths to image files
            image_root: Root directory for relative paths
            return_dummy_on_error: Whether to return dummy tensor on error
            
        Returns:
            List of preprocessed image tensors
        """
        processed_images = []
        
        for image_path in image_paths:
            pixel_values = self.load_image(
                image_path, 
                image_root, 
                return_dummy_on_error
            )
            processed_images.append(pixel_values)
        
        return processed_images
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        total = sum(self.stats.values())
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'success_rate': self.stats['processed'] / total,
            'error_rate': (self.stats['errors'] + self.stats['corrupted'] + self.stats['missing']) / total
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'processed': 0,
            'errors': 0,
            'corrupted': 0,
            'missing': 0
        }


def create_image_processor(image_size: int = 448,
                          max_num_patches: int = 12,
                          use_thumbnail: bool = True) -> ImageProcessor:
    """
    Convenience function to create image processor with default settings
    
    Args:
        image_size: Target size for each image patch
        max_num_patches: Maximum number of patches to generate
        use_thumbnail: Whether to add thumbnail patch
        
    Returns:
        Configured ImageProcessor instance
    """
    return ImageProcessor(
        image_size=image_size,
        max_num_patches=max_num_patches,
        use_thumbnail=use_thumbnail
    )


# Utility functions for compatibility with InternVL3 examples
def build_transform(input_size: int = 448) -> T.Compose:
    """Build transform pipeline (compatibility function)"""
    processor = ImageProcessor(image_size=input_size)
    return processor.transform


def load_image(image_file: str, 
               input_size: int = 448, 
               max_num: int = 12) -> torch.Tensor:
    """Load image following InternVL3 pattern (compatibility function)"""
    processor = ImageProcessor(
        image_size=input_size,
        max_num_patches=max_num,
        use_thumbnail=True
    )
    return processor.load_image(image_file)