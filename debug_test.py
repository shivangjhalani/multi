import logging
logging.basicConfig(level=logging.INFO)
import sys
import torch
sys.path.insert(0, '.')

from multimodal_coconut.model.multimodal_coconut import create_multimodal_coconut_model
from multimodal_coconut.config import create_config_from_template

# Create test setup
config = create_config_from_template('debug')
config.model_id = "OpenGVLab/InternVL3-1B-Pretrained"
device = "cuda" if torch.cuda.is_available() else "cpu"
config.torch_dtype = "bfloat16" if device == "cuda" else "float32"
config.image_size = 224
config.max_num_patches = 4
config.dynamic_preprocess = False

model, tokenizer = create_multimodal_coconut_model(config)
model = model.to(device)
model.eval()

# Test input
latent_token = "<|latent|>"
input_text = f"The color of the ball is {latent_token}."
input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)

print(f"Input text: {input_text}")
print(f"Input IDs: {input_ids}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
print(f"IMG_CONTEXT token ID: {getattr(model.base_model, 'img_context_token_id', 'NOT SET')}")

# Check if IMG_CONTEXT tokens are in the input
img_context_token_id = getattr(model.base_model, 'img_context_token_id', None)
if img_context_token_id is not None:
    has_img_context = (input_ids == img_context_token_id).any()
    print(f"Input contains IMG_CONTEXT tokens: {has_img_context}")
else:
    print("IMG_CONTEXT token ID not set")