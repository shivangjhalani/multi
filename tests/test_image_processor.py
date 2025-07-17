import pytest
import torch
from PIL import Image
from pathlib import Path

from multimodal_coconut.data.image_processor import ImageProcessor

@pytest.fixture
def image_processor():
    """Fixture for a default ImageProcessor."""
    return ImageProcessor(image_size=224, max_num_patches=6, use_thumbnail=True)

@pytest.fixture
def temp_image_dir(tmp_path):
    """Fixture to create a temporary directory with dummy images."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create a valid image
    valid_image = Image.new('RGB', (400, 300), color = 'red')
    valid_image.save(image_dir / "valid.jpg")
    
    # Create a corrupted image file
    corrupted_file = image_dir / "corrupted.jpg"
    with open(corrupted_file, 'w') as f:
        f.write("this is not an image")
        
    return image_dir

def test_image_processor_init(image_processor):
    """Test the initialization of the ImageProcessor."""
    assert image_processor.image_size == 224
    assert image_processor.max_num_patches == 6

def test_load_valid_image(image_processor, temp_image_dir):
    """Test loading a valid image file."""
    image_path = temp_image_dir / "valid.jpg"
    pixel_values = image_processor.load_image(str(image_path))
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.dim() == 4  # [num_patches, C, H, W]
    assert pixel_values.shape[1:] == (3, 224, 224)

def test_load_missing_image(image_processor):
    """Test that loading a missing image returns a dummy tensor."""
    pixel_values = image_processor.load_image("non_existent.jpg")
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 3, 224, 224) # Dummy image has 1 patch
    assert image_processor.get_stats()['missing'] == 1

def test_load_corrupted_image(image_processor, temp_image_dir):
    """Test that loading a corrupted image returns a dummy tensor."""
    image_path = temp_image_dir / "corrupted.jpg"
    pixel_values = image_processor.load_image(str(image_path))
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 3, 224, 224)
    assert image_processor.get_stats()['corrupted'] == 1

def test_dynamic_preprocessing(image_processor):
    """Test the dynamic preprocessing logic for splitting an image into patches."""
    # Test with a wide image (e.g., 16:9 aspect ratio)
    wide_image = Image.new('RGB', (1600, 900))
    patches = image_processor.dynamic_preprocess(wide_image)
    # based on the logic, it should find a 2x1 grid (ratio 2.0) for 16/9=1.77
    # plus a thumbnail
    assert len(patches) == 2 * 1 + 1 

    # Test with a tall image
    tall_image = Image.new('RGB', (900, 1600))
    patches = image_processor.dynamic_preprocess(tall_image)
    # 1x2 grid + thumbnail
    assert len(patches) == 1 * 2 + 1
    
    # Test with a square image
    square_image = Image.new('RGB', (500, 500))
    patches = image_processor.dynamic_preprocess(square_image)
    # 1x1 grid, no thumbnail
    assert len(patches) == 1

def test_batch_loading(image_processor, temp_image_dir):
    """Test loading a batch of images, including valid, missing, and corrupted."""
    image_paths = [
        str(temp_image_dir / "valid.jpg"),
        "non_existent.jpg",
        str(temp_image_dir / "corrupted.jpg")
    ]
    
    processed_batch = image_processor.load_images_batch(image_paths)
    assert len(processed_batch) == 3
    assert processed_batch[0].shape[1:] == (3, 224, 224) # Valid
    assert processed_batch[1].shape == (1, 3, 224, 224)   # Missing
    assert processed_batch[2].shape == (1, 3, 224, 224)   # Corrupted
    
    stats = image_processor.get_stats()
    assert stats['processed'] == 1
    assert stats['missing'] == 1
    assert stats['corrupted'] == 1