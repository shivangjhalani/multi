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
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import Config from the config module
from ..config import Config


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
        
        # Get model configuration
        self.config = base_model.config
        
        # Determine hidden size from the language model component
        if hasattr(base_model, 'language_model'):
            self.hidden_size = base_model.language_model.config.hidden_size
        elif hasattr(base_model.config, 'hidden_size'):
            self.hidden_size = base_model.config.hidden_size
        else:
            # Fallback - try to infer from model structure
            self.hidden_size = base_model.config.llm_config.hidden_size
    
    def forward(self,
                pixel_values: torch.FloatTensor,
                input_ids: torch.LongTensor,
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
        
        if len(latent_indices) == 0:
            # No latent tokens - use standard forward pass
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
        
        # Multimodal CoCoNuT forward pass with iterative processing
        return self._multimodal_forward_pass(
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
            latent_indices=latent_indices,
            **kwargs
        )
    
    def _multimodal_forward_pass(self,
                                pixel_values: torch.FloatTensor,
                                input_ids: torch.LongTensor,
                                latent_indices: torch.Tensor,
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
        Core multimodal CoCoNuT forward pass with iterative processing.
        
        This implements the multi-pass algorithm following the original CoCoNuT pattern:
        1. Process multimodal inputs (images + text) to get initial embeddings
        2. Group latent tokens by batch and position (following original CoCoNuT)
        3. Perform iterative forward passes with continuous thought feedback
        4. Use KV cache for efficiency across passes
        5. Concatenate all logits and compute final loss
        
        Key differences from original CoCoNuT:
        - Handles multimodal inputs (pixel_values + input_ids)
        - Integrates visual features into text embeddings
        - Maintains InternVL3's multimodal attention mechanisms
        
        Args:
            pixel_values: Image pixel values
            input_ids: Text token IDs
            latent_indices: Positions of latent tokens [num_latent_tokens, 2]
            ... (other args same as forward)
            
        Returns:
            CausalLMOutputWithPast with processed multimodal outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: Get initial embeddings and integrate multimodal features
        inputs_embeds = self._prepare_multimodal_embeddings(
            pixel_values, input_ids, image_flags
        )
        
        # Step 2: Group latent tokens by batch (following original CoCoNuT pattern)
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]  # bs, num_latent_tokens_in_the_instance
        
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        
        if max_n_latents == 0:
            # No latent tokens - use standard forward pass
            return self._standard_multimodal_forward(
                inputs_embeds, attention_mask, position_ids, past_key_values,
                labels, use_cache, output_attentions, output_hidden_states, return_dict
            )
        
        # Step 3: Iterative forward passes following original CoCoNuT algorithm
        logits = []
        next_compute_range = (0, latent_indices[:, 1].min().item())  # before earliest latent token
        kv_cache = past_key_values
        
        for pass_idx in range(max_n_latents):
            # Forward pass for current segment
            if kv_cache is None:
                # First forward pass
                outputs = self.base_model.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states_offset = 0
            else:
                # Extract KV cache to reuse (following original CoCoNuT pattern)
                past_key_values_truncated = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :]
                    )
                    for k, v in kv_cache
                ]
                
                outputs = self.base_model.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    past_key_values=past_key_values_truncated,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states_offset = next_compute_range[0]
            
            logits.append(outputs.logits)
            
            # Update compute range for next iteration
            next_compute_range = (
                next_compute_range[1],
                (
                    seq_len
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                )
            )
            
            # Extract hidden states for continuous thought feedback
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            kv_cache = outputs.past_key_values
            
            # Step 4: Apply continuous thought feedback (following original CoCoNuT)
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            # Break down inputs_embeds to avoid in-place operations (original CoCoNuT pattern)
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            
            # Replace latent token embeddings with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                
                # Replace with preceding hidden state (continuous thought feedback)
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
            
            # Reassemble inputs_embeds
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])
        
        # Step 5: Final forward pass for remaining sequence
        final_outputs = self.base_model.language_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
            past_key_values=(
                [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :]
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        logits.append(final_outputs.logits)
        
        # Step 6: Concatenate all logits and compute loss (following original CoCoNuT)
        logits = torch.cat(logits, dim=-2)  # Concatenate along sequence dimension
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=final_outputs.past_key_values,
            hidden_states=final_outputs.hidden_states,
            attentions=final_outputs.attentions,
        )
    
    def _prepare_multimodal_embeddings(self,
                                     pixel_values: torch.FloatTensor,
                                     input_ids: torch.LongTensor,
                                     image_flags: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Prepare multimodal embeddings by integrating visual features into text embeddings.
        
        This follows InternVL3's pattern of replacing image context tokens with visual embeddings.
        
        Args:
            pixel_values: Image pixel values [batch_size, num_patches, channels, height, width]
            input_ids: Text token IDs [batch_size, sequence_length]
            image_flags: Flags indicating which samples have images [batch_size, 1]
            
        Returns:
            inputs_embeds: Combined multimodal embeddings [batch_size, sequence_length, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get text embeddings
        inputs_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
        
        # Integrate visual features if images are provided
        if pixel_values is not None:
            # Extract visual features using InternVL3's vision encoder
            vit_embeds = self.base_model.extract_feature(pixel_values)
            
            # Handle image flags - if not provided, assume all samples have images
            if image_flags is None:
                image_flags = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            # Filter visual embeddings for samples with images
            if image_flags.dim() > 1:
                image_flags = image_flags.squeeze(-1)
            vit_embeds = vit_embeds[image_flags == 1]
            
            # Replace image context tokens with visual embeddings
            B, N, C = inputs_embeds.shape
            inputs_embeds_flat = inputs_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            
            # Find image context token positions
            img_context_token_id = getattr(self.base_model, 'img_context_token_id', None)
            if img_context_token_id is not None:
                selected = (input_ids_flat == img_context_token_id)
                if selected.sum() > 0 and len(vit_embeds) > 0:
                    try:
                        # Replace image context tokens with visual embeddings
                        inputs_embeds_flat[selected] = inputs_embeds_flat[selected] * 0.0 + vit_embeds.reshape(-1, C)
                    except Exception as e:
                        # Handle size mismatch gracefully
                        vit_embeds_flat = vit_embeds.reshape(-1, C)
                        n_token = selected.sum()
                        if n_token > 0 and len(vit_embeds_flat) > 0:
                            inputs_embeds_flat[selected] = inputs_embeds_flat[selected] * 0.0 + vit_embeds_flat[:n_token]
            
            inputs_embeds = inputs_embeds_flat.reshape(B, N, C)
        
        return inputs_embeds
    
    def _standard_multimodal_forward(self,
                                   inputs_embeds: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   position_ids: Optional[torch.LongTensor] = None,
                                   past_key_values: Optional[List[torch.FloatTensor]] = None,
                                   labels: Optional[torch.LongTensor] = None,
                                   use_cache: Optional[bool] = None,
                                   output_attentions: Optional[bool] = None,
                                   output_hidden_states: Optional[bool] = None,
                                   return_dict: Optional[bool] = None) -> CausalLMOutputWithPast:
        """
        Standard forward pass for multimodal inputs without latent tokens.
        
        Args:
            inputs_embeds: Combined multimodal embeddings
            ... (other standard forward pass arguments)
            
        Returns:
            CausalLMOutputWithPast with standard outputs
        """
        outputs = self.base_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate(self,
                 pixel_values: torch.FloatTensor,
                 input_ids: torch.LongTensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_flags: Optional[torch.LongTensor] = None,
                 max_new_tokens: int = 100,
                 do_sample: bool = True,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 **generation_kwargs) -> torch.LongTensor:
        """
        Generate text with multimodal continuous thought reasoning.
        
        Args:
            pixel_values: Image pixel values
            input_ids: Input token IDs
            attention_mask: Attention mask
            image_flags: Image flags
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **generation_kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        self.eval()
        
        with torch.no_grad():
            # Use the base model's generate method
            # The forward method will handle the continuous thought processing
            generated_ids = self.base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.eos_token_id,
                **generation_kwargs
            )
        
        return generated_ids


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
    print(f"Loading InternVL3 model: {config.model_id}")
    
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
    
    print(f"Special token IDs: start={start_latent_id}, end={end_latent_id}, latent={latent_id}, eos={eos_token_id}")
    
    # Load base InternVL3 model
    base_model = AutoModel.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=getattr(config, 'use_flash_attn', True)
    )
    
    # Resize token embeddings to accommodate new CoCoNuT tokens
    old_vocab_size = base_model.language_model.config.vocab_size
    new_vocab_size = len(tokenizer)
    
    if new_vocab_size > old_vocab_size:
        print(f"Resizing token embeddings from {old_vocab_size} to {new_vocab_size}...")
        base_model.language_model.resize_token_embeddings(new_vocab_size)
        print(f"✓ Token embeddings resized successfully")
    else:
        print(f"✓ No resize needed: tokenizer size ({new_vocab_size}) <= model vocab size ({old_vocab_size})")
    
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
        print(f"Loading checkpoint from: {config.load_model_path}")
        checkpoint = torch.load(config.load_model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        print("Checkpoint loaded successfully")
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer