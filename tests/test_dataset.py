import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset

from multimodal_coconut.data.dataset import MultimodalDataset, MultimodalCollator, get_multimodal_cot_latent_dataset
from multimodal_coconut.config.config import Config

@pytest.fixture
def tokenizer():
    """Fixture for a tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@pytest.fixture
def temp_data(tmp_path):
    """Fixture to create a temporary data directory with a dummy dataset and images."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image_dir = data_dir / "images"
    image_dir.mkdir()

    # Create dummy image
    dummy_image = Image.new('RGB', (100, 100), color = 'blue')
    dummy_image.save(image_dir / "test_image.jpg")

    # Create dummy data file
    dummy_data = [
        {
            "image_path": "test_image.jpg",
            "question": "What color is the image?",
            "steps": ["The image is a solid color.", "The color is blue."],
            "answer": "blue"
        }
    ]
    data_file = data_dir / "dataset.json"
    with open(data_file, 'w') as f:
        json.dump(dummy_data, f)
        
    return data_dir, data_file, image_dir

def test_multimodal_dataset_init(temp_data, tokenizer):
    """Test the initialization of the MultimodalDataset."""
    _, data_file, image_dir = temp_data
    dataset = MultimodalDataset(
        data_path=str(data_file),
        tokenizer=tokenizer,
        image_root=str(image_dir)
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert "pixel_values" in sample
    assert "question_tokenized" in sample
    # The HF dataset converts tensors to lists, so we convert back for the test
    pixel_values_tensor = torch.tensor(sample['pixel_values'])
    assert pixel_values_tensor.shape[1:] == (3, 448, 448)

def test_multimodal_collator(temp_data, tokenizer):
    """Test the MultimodalCollator for batching."""
    _, data_file, image_dir = temp_data
    tokenizer.pad_token_id = 0
    
    # Create a dataset and get a few samples
    dataset = MultimodalDataset(
        data_path=str(data_file),
        tokenizer=tokenizer,
        image_root=str(image_dir)
    )
    features = [dataset[0], dataset[0]] # Batch of 2 identical samples

    collator = MultimodalCollator(tokenizer=tokenizer)
    batch = collator(features)

    assert "pixel_values" in batch
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert batch["pixel_values"].shape[0] == batch["image_flags"].shape[0] * torch.tensor(features[0]['pixel_values']).shape[0]
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].dim() == 2

def test_get_multimodal_cot_latent_dataset(temp_data, tokenizer):
    """Test the replacement of reasoning steps with latent tokens."""
    _, data_file, image_dir = temp_data
    
    # Add special tokens for CoCoNuT
    special_tokens = {'additional_special_tokens': ['<latent>', '<start>', '<end>']}
    tokenizer.add_special_tokens(special_tokens)
    latent_id = tokenizer.convert_tokens_to_ids('<latent>')
    start_id = tokenizer.convert_tokens_to_ids('<start>')
    end_id = tokenizer.convert_tokens_to_ids('<end>')

    base_dataset = MultimodalDataset(
        data_path=str(data_file),
        tokenizer=tokenizer,
        image_root=str(image_dir)
    ).dataset
    
    configs = Config({
        "c_thought": 1,
        "max_latent_stage": 2,
        "uniform_prob": 0.0,
        "epochs_per_stage": 1
    })

    # Test stage 1: one reasoning step should be replaced
    staged_dataset = get_multimodal_cot_latent_dataset(
        scheduled_stage=1,
        base_dataset=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id
    )
    
    sample = staged_dataset[0]
    assert latent_id in sample['input_ids']
    
    # Count latent tokens
    num_latent_tokens = sample['input_ids'].count(latent_id)
    assert num_latent_tokens == 1 # c_thought * stage = 1 * 1

    # Test stage 2: two reasoning steps should be replaced
    staged_dataset_2 = get_multimodal_cot_latent_dataset(
        scheduled_stage=2,
        base_dataset=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id
    )
    
    sample_2 = staged_dataset_2[0]
    num_latent_tokens_2 = sample_2['input_ids'].count(latent_id)
    assert num_latent_tokens_2 == 2 # c_thought * stage = 1 * 2

# This requires PIL to be installed, so we import it inside the fixture
from PIL import Image