import torch
import pytest
from unittest.mock import MagicMock, PropertyMock

# Ensure the package is importable
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from transformers.modeling_outputs import CausalLMOutputWithPast

# --- Test Fixtures ---

@pytest.fixture
def mock_base_model():
    """Mocks the base InternVL model."""
    model = MagicMock()
    
    # Mock config
    mock_config = MagicMock()
    mock_config.use_return_dict = True
    type(model).config = PropertyMock(return_value=mock_config)
    
    # Mock hidden size
    type(model).hidden_size = PropertyMock(return_value=64)
    
    # Mock embeddings
    mock_embeddings = MagicMock()
    mock_embeddings.weight = torch.randn(100, 64)
    model.get_input_embeddings.return_value = mock_embeddings
    
    # Mock forward pass
    def forward_mock(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        inputs_embeds = kwargs.get('inputs_embeds')
        
        if inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = 1, 1

        return CausalLMOutputWithPast(
            logits=torch.randn(batch_size, seq_len, 100),
            past_key_values=MagicMock(),
            hidden_states=[torch.randn(batch_size, seq_len, 64)]
        )
    
    model.forward.side_effect = forward_mock
    model.return_value = model.forward.side_effect
    
    return model

@pytest.fixture
def coconut_model(mock_base_model):
    """Creates a MultimodalCoconut model instance with a mocked base model."""
    return MultimodalCoconut(
        base_model=mock_base_model,
        latent_token_id=1,
        start_latent_id=2,
        end_latent_id=3,
        eos_token_id=4
    )

# --- Test Cases ---

def test_model_initialization(coconut_model):
    """Test if the model initializes correctly."""
    assert isinstance(coconut_model, MultimodalCoconut)
    assert coconut_model.latent_token_id == 1
    assert coconut_model.hidden_size == 64

def test_standard_forward_pass_no_latents(coconut_model):
    """Test the standard forward pass when no latent tokens are present."""
    input_ids = torch.randint(10, 100, (1, 10))
    pixel_values = torch.randn(1, 3, 224, 224)
    
    outputs = coconut_model.forward(input_ids=input_ids, pixel_values=pixel_values)
    
    assert 'logits' in outputs
    coconut_model.base_model.forward.assert_called_once()

def test_iterative_forward_pass_with_latents(coconut_model, mock_base_model):
    """Test the iterative forward pass with latent tokens."""
    # input_ids: [cls, a, b, latent, c, d, eos]
    input_ids = torch.tensor([[5, 10, 11, 1, 12, 13, 4]])
    pixel_values = torch.randn(1, 3, 224, 224)
    attention_mask = torch.ones_like(input_ids)

    outputs = coconut_model.forward(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )

    # The forward pass should be called twice: once for the segment before the latent token,
    # and once for the segment after.
    assert mock_base_model.call_count == 2
    assert 'logits' in outputs

def test_causal_kv_cache_usage(coconut_model, mock_base_model):
    """Ensure past_key_values are passed between iterative steps."""
    input_ids = torch.tensor([[5, 10, 1, 11, 1, 12]]) # two latent tokens
    pixel_values = torch.randn(1, 3, 224, 224)
    
    coconut_model.forward(input_ids=input_ids, pixel_values=pixel_values)
    
    # Three calls total, and the second and third should receive `past_key_values`
    assert mock_base_model.call_count == 3
    
    # Check second call
    args, kwargs = mock_base_model.call_args_list[1]
    assert 'past_key_values' in kwargs
    assert kwargs['past_key_values'] is not None

    # Check third call
    args, kwargs = mock_base_model.call_args_list[2]
    assert 'past_key_values' in kwargs
    assert kwargs['past_key_values'] is not None

def test_dynamic_visual_processing(coconut_model, mock_base_model):
    """Verify pixel_values are passed in every call during iterative pass."""
    input_ids = torch.tensor([[5, 10, 1, 11, 12]])
    pixel_values = torch.randn(1, 3, 224, 224)
    
    coconut_model.forward(input_ids=input_ids, pixel_values=pixel_values)
    
    # Check that pixel_values were passed in both calls
    for call in mock_base_model.call_args_list:
        _, kwargs = call
        assert 'pixel_values' in kwargs
        assert kwargs['pixel_values'] is not None

def test_attention_mask_and_position_ids_slicing(coconut_model, mock_base_model):
    """
    Test if attention_mask and position_ids are correctly handled in the iterative forward pass.
    This test is expected to fail with the current implementation.
    """
    input_ids = torch.tensor([[5, 10, 1, 11, 12, 4]]) # seq_len = 6
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)

    coconut_model.forward(
        input_ids=input_ids,
        pixel_values=torch.randn(1, 3, 224, 224),
        attention_mask=attention_mask,
        position_ids=position_ids
    )

    # First call: process up to latent token (3 tokens)
    call1_kwargs = mock_base_model.call_args_list[0][1]
    # The current buggy implementation passes the full-length attention_mask
    assert call1_kwargs['attention_mask'].shape[1] != input_ids.shape[1]
    
    # Second call: process remaining tokens (3 tokens)
    call2_kwargs = mock_base_model.call_args_list[1][1]
    # The current buggy implementation passes the full-length position_ids
    assert call2_kwargs['position_ids'].shape[1] != input_ids.shape[1]

def test_generate_method_with_latents(coconut_model, mock_base_model):
    """Test the generate method with latent tokens."""
    input_ids = torch.tensor([[5, 10, 1, 11, 12, 4]])

    # The current generate is not fully implemented for iterative reasoning
    # and should be fixed. For now, let's confirm it runs and we can
    # add a more specific test once we fix it.
    generated_ids = coconut_model.generate(
        pixel_values=torch.randn(1, 3, 224, 224),
        input_ids=input_ids,
        generation_config={'max_new_tokens': 5}
    )
    assert generated_ids.shape[1] == input_ids.shape[1] + 5


if __name__ == "__main__":
    pytest.main([__file__])