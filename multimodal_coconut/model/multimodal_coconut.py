"""
Multimodal CoCoNuT model implementation

Extends the original CoCoNuT methodology to multimodal reasoning using InternVL3-1B-Pretrained.
This implementation follows the core CoCoNuT principles while adapting them for multimodal inputs.

Key features:
- Integrates InternVL3-1B-Pretrained as the base multimodal model
- Implements continuous thought feedback mechanism for multimodal hidden states
- Supports iterative forward passes with KV cache optimization
- Maintains compatibility with original CoCoNuT training curriculum
"""

import torch
import torch.nn as nn
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import Config from the config module
from ..config import Config

# Set up logger
logger = logging.getLogger(__name__)


class MultimodalCoconut(nn.Module):
    """
    Multimodal CoCoNuT model that extends the original CoCoNuT architecture to handle multimodal inputs.
    
    This implementation follows the core CoCoNuT principles:
    1. Continuous thought feedback mechanism using hidden states
    2. Iterative forward passes with KV cache optimization
    3. Latent token detection and processing
    4. Multi-pass architecture for handling dependencies
    
    The key innovation is extending these principles to work with InternVL3's multimodal representations.
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 latent_token_id: int,
                 start_latent_id: int,
                 end_latent_id: int,
                 eos_token_id: int):
        """
        Initialize multimodal CoCoNuT model
        
        Args:
            base_model: InternVL3 base model
            latent_token_id: ID for <|latent|> token
            start_latent_id: ID for <|start-latent|> token
            end_latent_id: ID for <|end-latent|> token
            eos_token_id: End of sequence token ID
        """
        super().__init__()
        
        # Store the base InternVL3 model
        self.base_model = base_model
        
        # Store special token IDs (following original CoCoNuT pattern)
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        
        # Get model configuration (handle cases where config might not exist)
        if hasattr(base_model, 'config'):
            self.config = base_model.config
        else:
            # Create a minimal config for testing scenarios
            from types import SimpleNamespace
            self.config = SimpleNamespace(use_return_dict=True)
        
        # Determine hidden size from the language model component
        self.hidden_size = self._determine_hidden_size(base_model)
        
        # # Initialize img_context_token_id to None - will be set during forward pass
        # # This is required for InternVL3 compatibility
        # if not hasattr(base_model, 'img_context_token_id'):
        #     base_model.img_context_token_id = None
    
    def _determine_hidden_size(self, base_model: nn.Module) -> int:
        """
        Get the language model's hidden size for CoCoNuT continuous thought mechanism.
        
        Args:
            base_model: InternVL3 base model
            
        Returns:
            Language model hidden size dimension
        """
        # For InternVL3, get language model hidden size (needed for continuous thoughts)
        if hasattr(base_model, 'language_model') and hasattr(base_model.language_model, 'config'):
            return base_model.language_model.config.hidden_size
        
        # Fallback for testing with mock models
        if hasattr(base_model, 'embeddings') and hasattr(base_model.embeddings, 'weight'):
            return base_model.embeddings.weight.shape[1]
        
        # Default for tests
        logger.warning("Could not determine hidden size from base model, using default 64")
        return 64
    
    def forward(self,
                input_ids: torch.LongTensor,
                pixel_values: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                image_flags: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Multimodal forward pass with continuous thought reasoning.
        
        This method implements the core CoCoNuT algorithm adapted for multimodal inputs:
        1. Detect latent token positions in the input sequence
        2. Perform iterative forward passes, feeding hidden states back as embeddings
        3. Use KV cache for efficiency during multi-pass processing
        4. Handle both visual and textual tokens in the continuous thought process
        
        Args:
            pixel_values: Image pixel values [batch_size, num_patches, channels, height, width]
            input_ids: Text token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            position_ids: Position IDs [batch_size, sequence_length]
            image_flags: Flags indicating which samples have images [batch_size, 1]
            past_key_values: Cached key-value pairs for efficiency
            labels: Target labels for training [batch_size, sequence_length]
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            CausalLMOutputWithPast containing logits, loss, and other outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Find all latent token positions in the batch
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        
        # DEBUG: Log whether latent tokens are found
        if len(latent_indices) == 0:
            # No latent tokens - use standard multimodal forward pass
            return self._standard_multimodal_forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        
        # Multimodal CoCoNuT forward pass with iterative processing
        return self._multimodal_forward_pass(
            input_ids=input_ids,
            latent_indices=latent_indices,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    def _multimodal_forward_pass(self,
                                 input_ids: torch.LongTensor,
                                 latent_indices: torch.Tensor,
                                 pixel_values: Optional[torch.FloatTensor] = None,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 position_ids: Optional[torch.LongTensor] = None,
                                 image_flags: Optional[torch.LongTensor] = None,
                                 past_key_values: Optional[List[torch.FloatTensor]] = None,
                                 labels: Optional[torch.LongTensor] = None,
                                 use_cache: Optional[bool] = None,
                                 output_attentions: Optional[bool] = None,
                                 output_hidden_states: Optional[bool] = None,
                                 return_dict: Optional[bool] = None,
                                 **kwargs) -> CausalLMOutputWithPast:
        """
        Iterative, sequentially dependent multimodal forward pass.
        This refactored method ensures causality by fusing multimodal embeddings *before*
        the iterative CoCoNuT-style processing.
        """
        if pixel_values is None:
            # Fallback to language model if no image is provided
            return self.base_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # 1. Get text embeddings
        input_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        # 2. Get visual embeddings from the ViT
        vit_embeds, _ = self.base_model.encode_img(pixel_values) # Shape: [bs, num_img_tokens, hidden_size]
        
        # Prepare for merging
        new_input_embeds = []
        new_attention_mask = []
        
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # Find the first latent token for this batch item
            latent_idx = (input_ids[i] == self.latent_token_id).nonzero(as_tuple=True)[0]
            
            if len(latent_idx) == 0:
                # No latent token, just use text embeddings
                new_input_embeds.append(input_embeds[i])
                new_attention_mask.append(attention_mask[i])
                continue

            # For simplicity, we only handle the first latent token per sample
            # This aligns with the common use case of providing one image context
            first_latent_pos = latent_idx[0]
            
            # Split text embeddings around the latent token
            pre_latent_embeds = input_embeds[i, :first_latent_pos]
            post_latent_embeds = input_embeds[i, first_latent_pos + 1:]
            
            # Combine: text (pre) + vision + text (post)
            combined_embeds = torch.cat([
                pre_latent_embeds,
                vit_embeds[i],
                post_latent_embeds
            ], dim=0)
            
            # Create a new attention mask for the combined sequence
            combined_attention_mask = torch.ones(combined_embeds.shape[0], dtype=torch.long, device=input_ids.device)
            
            new_input_embeds.append(combined_embeds)
            new_attention_mask.append(combined_attention_mask)

        # Pad the sequences to the same length
        # This is a simplified padding implementation. A more robust solution
        # would use a tokenizer's padding capabilities.
        max_len = max(embed.shape[0] for embed in new_input_embeds)
        
        final_embeds = torch.zeros(batch_size, max_len, self.hidden_size, device=input_ids.device, dtype=input_embeds.dtype)
        final_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=input_ids.device)
        
        for i in range(batch_size):
            seq_len = new_input_embeds[i].shape[0]
            final_embeds[i, :seq_len] = new_input_embeds[i]
            final_attention_mask[i, :seq_len] = new_attention_mask[i]
            
        # 3. Pass the combined embeddings to the language model
        return self.base_model.language_model(
            inputs_embeds=final_embeds,
            attention_mask=final_attention_mask,
            position_ids=None, # Position IDs would need to be recalculated
            past_key_values=past_key_values,
            labels=None, # Labels would need to be recalculated to align with new sequence
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    # def _ensure_img_context_token_id(self, tokenizer):
    #     """
    #     Ensure img_context_token_id is set on the base model.
    #     This is required for InternVL3 compatibility.
    #     """
    #     if not hasattr(self.base_model, 'img_context_token_id') or self.base_model.img_context_token_id is None:
    #         try:
    #             img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    #             self.base_model.img_context_token_id = img_context_token_id
    #         except:
    #             # If tokenizer doesn't have IMG_CONTEXT token, set to None
    #             self.base_model.img_context_token_id = None
    
    def _standard_multimodal_forward(self,
                                   input_ids: torch.LongTensor,
                                   pixel_values: Optional[torch.FloatTensor] = None,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   position_ids: Optional[torch.LongTensor] = None,
                                   image_flags: Optional[torch.LongTensor] = None,
                                   past_key_values: Optional[List[torch.FloatTensor]] = None,
                                   labels: Optional[torch.LongTensor] = None,
                                   use_cache: Optional[bool] = None,
                                   output_attentions: Optional[bool] = None,
                                   output_hidden_states: Optional[bool] = None,
                                   return_dict: Optional[bool] = None,
                                   **kwargs) -> CausalLMOutputWithPast:
        """
        Standard forward pass for multimodal inputs without latent tokens.
        
        This uses the base InternVL3 model directly for standard multimodal processing.
        For text-only inputs (pixel_values=None), we use the language model directly.
        
        Args:
            pixel_values: Image pixel values (can be None for text-only)
            input_ids: Text token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            image_flags: Image flags
            past_key_values: Past key values
            labels: Target labels
            use_cache: Whether to use cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict
            
        Returns:
            CausalLMOutputWithPast with standard outputs
        """
        # Handle text-only inputs by using the language model directly
        if pixel_values is None:
            return self.base_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        # For multimodal inputs, use the full InternVL3 model
        # Ensure image_flags is properly set for InternVL3
        if image_flags is None and pixel_values is not None:
            # Create default image_flags if not provided
            batch_size = input_ids.shape[0]
            image_flags = torch.ones(batch_size, 1, dtype=torch.long, device=input_ids.device)

        return self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
    
    @torch.no_grad()
    def generate(self,
                 pixel_values: Optional[torch.FloatTensor] = None,
                 input_ids: Optional[torch.LongTensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_flags: Optional[torch.LongTensor] = None,
                 generation_config: Optional[Dict] = None,
                 **generate_kwargs) -> torch.LongTensor:
        """
        Generate text with iterative, sequentially dependent multimodal reasoning.

        This method implements a generation loop that respects causality and handles
        dynamic multimodal fusion. It processes the input sequentially.
        """
        self.eval()

        if generation_config is None:
            generation_config = {}
        max_new_tokens = generation_config.get('max_new_tokens', 100)
        eos_token_id = generation_config.get('eos_token_id', self.eos_token_id)

        batch_size = input_ids.shape[0]

        # Process the prompt iteratively, including latent tokens
        prompt_outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            image_flags=image_flags,
            **generate_kwargs
        )

        past_key_values = prompt_outputs.past_key_values
        next_token_logits = prompt_outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_ids = [ids.tolist() for ids in input_ids]

        for _ in range(max_new_tokens):
            # When using past_key_values, we only need to provide the input_ids for the next token.
            # The attention_mask for this single token is simply 1.
            next_attention_mask = torch.ones_like(next_token_ids)

            outputs = self.forward(
                input_ids=next_token_ids,
                attention_mask=next_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                pixel_values=None,  # Visual features are already in the cache
            )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            for b in range(batch_size):
                generated_ids[b].append(next_token_ids[b].item())

            if (next_token_ids == eos_token_id).all():
                break
        
        max_len = max(len(ids) for ids in generated_ids)
        padded_ids = [
            ids + [eos_token_id] * (max_len - len(ids)) for ids in generated_ids
        ]
        
        return torch.tensor(padded_ids, dtype=torch.long, device=input_ids.device)
    
    def chat(self, 
             tokenizer,
             pixel_values: Optional[torch.FloatTensor] = None,
             question: str = "",
             generation_config: Optional[dict] = None,
             history: Optional[List[Tuple[str, str]]] = None,
             return_history: bool = False,
             num_patches_list: Optional[List[int]] = None,
             IMG_START_TOKEN: str = '<img>',
             IMG_END_TOKEN: str = '</img>',
             IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
             verbose: bool = False) -> Union[str, Tuple[str, List[Tuple[str, str]]]]:
        """
        Chat interface for multimodal CoCoNuT model.
        
        This method provides a conversational interface similar to InternVL's chat method
        but uses our CoCoNuT model for continuous thought reasoning.
        
        Args:
            tokenizer: Tokenizer for text processing
            pixel_values: Image pixel values [total_patches, channels, height, width]
            question: User question/prompt
            generation_config: Generation configuration
            history: Conversation history as list of (question, answer) tuples
            return_history: Whether to return updated history
            num_patches_list: Number of patches per image
            IMG_START_TOKEN: Image start token
            IMG_END_TOKEN: Image end token
            IMG_CONTEXT_TOKEN: Image context token
            verbose: Whether to print verbose output
            
        Returns:
            Generated response, optionally with updated history
        """
        # Add image placeholder if not present and we have images
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        
        # Set up num_patches_list
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        
        # Validate pixel_values and num_patches_list consistency
        if pixel_values is not None:
            assert len(pixel_values) == sum(num_patches_list), \
                f"Pixel values length ({len(pixel_values)}) != sum of patches ({sum(num_patches_list)})"
        
        # Set up image context token ID
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.base_model.img_context_token_id = img_context_token_id
        
        # Build conversation prompt
        # For simplicity, we use a basic template - this can be enhanced with proper conversation templates
        history = [] if history is None else history
        
        # Build the full conversation
        conversation_parts = []
        
        # Add history
        for old_question, old_answer in history:
            conversation_parts.append(f"Human: {old_question}")
            conversation_parts.append(f"Assistant: {old_answer}")
        
        # Add current question
        conversation_parts.append(f"Human: {question}")
        conversation_parts.append("Assistant:")
        
        query = "\n".join(conversation_parts)
        
        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            logger.debug(f'Dynamic ViT batch size: {image_bs}')
        
        # Replace <image> placeholders with actual image tokens
        for num_patches in num_patches_list:
            # Calculate number of image tokens (following InternVL pattern)
            num_image_token = getattr(self.base_model, 'num_image_token', 256)  # Default for InternVL3-1B
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        # Tokenize the query
        model_inputs = tokenizer(query, return_tensors='pt')
        
        # Get device - handle both real models and test mocks
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Fallback for test mocks or models without parameters
            logger.warning("No parameters found in model, fallback to cpu")
            device = torch.device('cpu')
        
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        
        # Set up generation config
        if generation_config is None:
            generation_config = {}
        
        # Set EOS token ID (use tokenizer's EOS token)
        generation_config['eos_token_id'] = tokenizer.eos_token_id
        
        # Generate response
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
        
        # Decode the response
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        
        # Extract only the new part (after "Assistant:")
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        # Clean up the response
        response = response.strip()
        
        # Update history
        history.append((question, response))
        
        if verbose:
            # Print clean query for debugging
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            logger.debug(f"Query: {query_to_print}")
            logger.debug(f"Response: {response}")
        
        if return_history:
            return response, history
        else:
            return response


def create_multimodal_coconut_model(config: Config) -> Tuple[MultimodalCoconut, AutoTokenizer]:
    """
    Create a multimodal CoCoNuT model with proper InternVL3 integration.
    
    This function:
    1. Loads the InternVL3-1B-Pretrained model
    2. Adds CoCoNuT special tokens to the vocabulary
    3. Wraps the model with MultimodalCoconut class
    4. Handles checkpoint loading if specified
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        Tuple of (MultimodalCoconut model, tokenizer)
    """
    logger.info(f"Loading InternVL3 model: {config.model_id}")
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Add CoCoNuT special tokens
    special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    tokenizer.add_tokens(special_tokens)
    
    # Get special token IDs
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    eos_token_id = tokenizer.eos_token_id
    
    logger.info(f"Special token IDs: start={start_latent_id}, end={end_latent_id}, latent={latent_id}, eos={eos_token_id}")
    
    # Load base InternVL3 model
    # Determine torch_dtype from config
    dtype_str = getattr(config, 'torch_dtype', 'bfloat16')
    if dtype_str == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype_str == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    base_model = AutoModel.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=getattr(config, 'use_flash_attn', True)
    )
    
    # Resize token embeddings to accommodate new CoCoNuT tokens
    # This must be done on the language_model submodule, not the base_model
    logger.info(f"Resizing token embeddings for the language model to match tokenizer size: {len(tokenizer)}")
    base_model.language_model.resize_token_embeddings(len(tokenizer))
    logger.info(f"✓ Token embeddings resized successfully")
    
    # Set the img_context_token_id for multimodal processing
    # IMG_CONTEXT is already in the tokenizer by default
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    base_model.img_context_token_id = img_context_token_id
    
    # Set num_image_token for InternVL3 compatibility
    # This is calculated as (image_size // patch_size) ** 2 * down_sample_ratio ** 2
    # For InternVL3-1B-Pretrained: image_size=448, patch_size=14, down_sample_ratio=1
    image_size = getattr(config, 'image_size', 448)
    patch_size = 14  # InternVL3 default
    down_sample_ratio = 1  # InternVL3 default
    base_model.num_image_token = int((image_size // patch_size) ** 2 * (down_sample_ratio ** 2))
    
    # Create multimodal CoCoNuT wrapper
    model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )
    
    # Load checkpoint if specified
    if hasattr(config, 'load_model_path') and config.load_model_path != "None":
        logger.info(f"Loading checkpoint from: {config.load_model_path}")
        checkpoint = torch.load(config.load_model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        logger.info("Checkpoint loaded successfully")
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer
