import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig

from multimodal_coconut.model.multimodal_coconut import MultimodalCoconut
from multimodal_coconut.data.dataset import MultimodalCollator

class MockInternVLConfig(PretrainedConfig):
    def __init__(self, hidden_size=64, vocab_size=1000, **kwargs):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

class MockLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, inputs_embeds, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast
        output = self.linear(inputs_embeds)
        return CausalLMOutputWithPast(logits=output, hidden_states=[inputs_embeds])

    def get_input_embeddings(self):
        return self.embeddings

class MockInternVLModel(PreTrainedModel):
    config_class = MockInternVLConfig

    def __init__(self, config):
        super().__init__(config)
        self.language_model = MockLanguageModel(config)

    def forward(self, inputs_embeds, **kwargs):
        return self.language_model(inputs_embeds, **kwargs)
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
        
    def extract_feature(self, pixel_values):
        # Return a dummy tensor of the correct shape
        num_patches = pixel_values.shape[0]
        return torch.randn(num_patches, self.config.hidden_size)


@pytest.fixture
def mock_model_and_tokenizer():
    """Fixture for a mock model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<IMG_CONTEXT>"]
    tokenizer.add_tokens(special_tokens)
    
    # Use the tokenizer's vocab size for the mock model config
    config = MockInternVLConfig(vocab_size=len(tokenizer))
    base_model = MockInternVLModel(config)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    
    model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=eos_id
    )
    model.base_model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    
    return model, tokenizer

def test_coconut_forward_pass(mock_model_and_tokenizer):
    """Test the CoCoNuT forward pass with latent tokens."""
    model, tokenizer = mock_model_and_tokenizer
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    
    # Create a batch with latent tokens
    input_ids = torch.tensor([
        [101, 10, 20, latent_id, 30, 102],
        [101, 40, latent_id, 50, 60, 102]
    ], dtype=torch.long)
    
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(2, 1, 3, 224, 224) # 2 samples, 1 patch each
    image_flags = torch.ones(2, 1, dtype=torch.long)

    # The model's forward pass should now execute with the new logging
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        image_flags=image_flags,
        labels=input_ids
    )

    assert outputs.loss is not None
    assert outputs.logits.shape == (2, 6, len(tokenizer))