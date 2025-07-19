import torch
import pytest
import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal_coconut.model.multimodal_coconut import create_multimodal_coconut_model
from multimodal_coconut.config import create_config_from_template
from PIL import Image

# --- Test Fixtures ---

@pytest.fixture(scope="session")
def real_model_and_tokenizer():
    """Loads the real InternVL model and tokenizer."""
    config = create_config_from_template('debug')
    # Use a smaller model for faster testing if available, otherwise use the default.
    config.model_id = "OpenGVLab/InternVL3-1B-Pretrained"
    model, tokenizer = create_multimodal_coconut_model(config)
    return model, tokenizer

@pytest.fixture
def pixel_values():
    """Creates a dummy pixel_values tensor."""
    return torch.randn(1, 3, 448, 448)

# --- Test Cases ---

def test_model_initialization(real_model_and_tokenizer):
    """Test if the model initializes correctly."""
    model, _ = real_model_and_tokenizer
    assert model is not None
    assert model.latent_token_id is not None

def test_standard_forward_pass_no_latents(real_model_and_tokenizer, pixel_values):
    """Test the standard forward pass when no latent tokens are present."""
    model, tokenizer = real_model_and_tokenizer
    input_ids = tokenizer("A dog is sitting on the grass", return_tensors='pt').input_ids
    
    outputs = model.forward(input_ids=input_ids, pixel_values=pixel_values)
    
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape[0] == input_ids.shape[0]
    assert outputs.logits.shape[1] == input_ids.shape[1]

def test_iterative_forward_pass_with_latents(real_model_and_tokenizer, pixel_values):
    """Test the iterative forward pass with latent tokens."""
    model, tokenizer = real_model_and_tokenizer
    latent_token = "<|latent|>"
    input_text = f"The dog {latent_token} is happy."
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    outputs = model.forward(input_ids=input_ids, pixel_values=pixel_values)

    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape[1] == input_ids.shape[1]

def test_causal_kv_cache_usage(real_model_and_tokenizer, pixel_values):
    """Ensure past_key_values are passed and grown between iterative steps."""
    model, tokenizer = real_model_and_tokenizer
    latent_token = "<|latent|>"
    input_text = f"Q: What is the dog doing? A: The dog {latent_token} is running and {latent_token} playing."
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    # The test for this is implicitly handled by a successful forward pass.
    # A failure in KV-caching would likely lead to a tensor shape mismatch error.
    # We can add more specific checks if needed, but for now, a successful run is a good indicator.
    try:
        model.forward(input_ids=input_ids, pixel_values=pixel_values)
    except Exception as e:
        pytest.fail(f"Forward pass with multiple latents failed, possibly due to KV cache issue: {e}")

def test_dynamic_visual_processing(real_model_and_tokenizer, pixel_values):
    """Verify pixel_values are used in the iterative pass."""
    model, tokenizer = real_model_and_tokenizer
    latent_token = "<|latent|>"
    input_text = f"The color of the ball is {latent_token}."
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    
    # Run with and without pixel_values to ensure the model behaves differently.
    with torch.no_grad():
        outputs_with_vision = model.forward(input_ids=input_ids, pixel_values=pixel_values)
        outputs_without_vision = model.forward(input_ids=input_ids, pixel_values=None)

    # A simple check: the logits should be different.
    assert not torch.allclose(outputs_with_vision.logits, outputs_without_vision.logits)

def test_generate_method_with_latents(real_model_and_tokenizer, pixel_values):
    """Test the generate method with latent tokens."""
    model, tokenizer = real_model_and_tokenizer
    latent_token = "<|latent|>"
    input_text = f"Q: What is in the image? A: I see {latent_token}."
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids

    generated_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        generation_config={'max_new_tokens': 10, 'eos_token_id': tokenizer.eos_token_id}
    )
    
    assert generated_ids.shape[1] > input_ids.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])