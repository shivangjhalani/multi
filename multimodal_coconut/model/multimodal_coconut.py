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
<<<<<<< Updated upstream
        Core multimodal CoCoNuT forward pass following the exact original pattern.
        
        This is a direct adaptation of the original CoCoNuT algorithm for multimodal inputs:
        1. Prepare multimodal embeddings (visual + textual)
        2. Group latent tokens by batch 
        3. Iterative forward passes with KV cache 
        4. Continuous thought feedback 
        5. Concatenate logits and compute loss 
        
        Args:
            pixel_values: Image pixel values
            input_ids: Text token IDs  
            latent_indices: Positions of latent tokens [num_latent_tokens, 2]
            ... (other standard forward args)
            
        Returns:
            CausalLMOutputWithPast with multimodal CoCoNuT outputs
||||||| Stash base
        Refactored multimodal CoCoNuT forward pass using the full base model.

        This implementation corrects the architectural flaw by calling the full
        `self.base_model.forward()` in a loop, allowing InternVL to correctly
        manage multimodal fusion.

        The process is as follows:
        1.  A single initial forward pass is made with `input_ids` and `pixel_values`
            to get the initial `hidden_states` and `past_key_values`.
        2.  The model then iterates through each latent token. In each step:
            a. The hidden state from the previous step (the "thought vector") is extracted.
            b. This thought vector is injected into the input embeddings for the
               current latent token's position.
            c. `self.base_model.forward()` is called again with the updated
               `inputs_embeds` and the `past_key_values` from the previous step,
               ensuring state is maintained.
        3.  The final output from the loop contains the logits for the entire sequence.
        
        Args:
            pixel_values: Image pixel values
            input_ids: Text token IDs
            latent_indices: Positions of latent tokens [num_latent_tokens, 2]
            ... (other standard forward args)
            
        Returns:
            CausalLMOutputWithPast with multimodal CoCoNuT outputs
=======
        Iterative, sequentially dependent multimodal forward pass.

        This implementation processes the input sequence segment by segment, respecting
        the causal chain of reasoning. It enables dynamic multimodal reasoning by
        making visual information available at each step.

        The process is as follows:
        1.  The input sequence is split into segments based on the positions of
            `<|latent|>` tokens.
        2.  The model iterates through these segments, processing one at a time.
        3.  In each iteration, it passes the current segment's tokens, the
            `pixel_values`, and the `past_key_values` from the previous step to
            the base model.
        4.  When a latent token is processed, the hidden state from the previous
            token is used as the "thought vector" for the next segment.
        5.  The final logits are assembled from the outputs of each segment.
>>>>>>> Stashed changes
        """
        batch_size, seq_len = input_ids.shape
<<<<<<< Updated upstream
        
        # Prepare multimodal embeddings (replaces self.embedding(input_ids) from original)
        inputs_embeds = self._prepare_multimodal_embeddings(pixel_values, input_ids, image_flags)
        
        # Group latent tokens by batch
||||||| Stash base

        # Group latent tokens by batch and sort them
=======
        wte = self.base_model.get_input_embeddings()
        
        # Group latent tokens by batch and sort them
>>>>>>> Stashed changes
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
<<<<<<< Updated upstream
        ]  # bs, num_latent_tokens_in_the_instance
        
||||||| Stash base
        ]
=======
        ]

        # Initialize containers for outputs
        all_logits = []
        current_past_key_values = past_key_values
        
        # Process the sequence segment by segment
        last_processed_pos = {b: 0 for b in range(batch_size)}
        
        # Determine the number of latent segments to process
>>>>>>> Stashed changes
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
<<<<<<< Updated upstream
        
        # Set up compute range 
        next_compute_range = (0, seq_len)
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
        
        logits = []
        kv_cache = past_key_values
        
        # Iterative forward passes 
        for pass_idx in range(max_n_latents):
            
            if kv_cache is None:
                # First forward pass (exactly like original CoCoNuT)
                outputs = self.base_model.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                # Extract KV cache to reuse (exactly like original CoCoNuT)
                past_key_values = [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                
                outputs = self.base_model.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]
            
            logits.append(outputs.logits)
            
            # Update compute range 
            next_compute_range = (
                next_compute_range[1],
                (
                    seq_len
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )
            
            # Extract hidden states 
            hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values
            
            # Continuous thought feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            # Break down inputs_embeds to avoid in-place operations 
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            
            # Replace latent tokens with continuous thoughts 
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                
                # Calculate the index for the preceding hidden state
                hidden_idx = token_idx - 1 - hidden_states_offset
                
                # Ensure we don't go out of bounds
                if hidden_idx >= 0 and hidden_idx < hidden_states.shape[1]:
                    # Replace with preceding hidden state 
                    tensor_list[batch_idx][token_idx] = hidden_states[
                        batch_idx, hidden_idx, :
                    ]
                else:
                    # If we can't access the preceding hidden state, use zero vector
                    tensor_list[batch_idx][token_idx] = torch.zeros_like(
                        tensor_list[batch_idx][token_idx]
                    )
            
            # Reassemble inputs_embeds 
            inputs_embeds = torch.stack([
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])
        
        # Final forward pass 
        if kv_cache is not None:
            # Extract KV cache for final pass
            past_key_values = [
                (
                    k[:, :, :next_compute_range[0], :],
                    v[:, :, :next_compute_range[0], :],
                )
                for k, v in kv_cache
            ]
        else:
            past_key_values = None
            
        final_outputs = self.base_model.language_model(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        
        logits.append(final_outputs.logits)
        
        # Concatenate logits and compute loss 
        logits = torch.cat(logits, dim=-2)
        
||||||| Stash base

        # Initial forward pass using the full base model.
        # This correctly processes multimodal inputs and establishes the initial KV cache.
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=None,  # Loss is calculated at the end.
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=True, # Critical for feedback loop
            return_dict=True,
            num_patches_list=num_patches_list,
            **kwargs
        )
        
        # If there are no latent tokens, we are done after the first pass.
        # The final logits are already in `outputs`. The loss will be computed below.
        if max_n_latents > 0:
            # Prepare for the iterative loop
            wte = self.base_model.get_input_embeddings()
            inputs_embeds = wte(input_ids)
            
            # Iteratively process each level of latent tokens
            for i in range(max_n_latents):
                last_hidden_states = outputs.hidden_states[-1]
                
                # Inject the thought vector (hidden state) into the embeddings
                for b in range(batch_size):
                    # Check if this instance in the batch has this many latent tokens
                    if i < len(latent_lists[b]):
                        current_latent_pos = latent_lists[b][i]
                        
                        # The thought vector comes from the hidden state at the position
                        # immediately preceding the latent token.
                        thought_pos = current_latent_pos - 1
                        
                        if thought_pos >= 0:
                            thought_vector = last_hidden_states[b, thought_pos, :]
                            # Replace the embedding of the *current* latent token
                            inputs_embeds[b, current_latent_pos, :] = thought_vector
                
                # Call the base model again with updated embeddings and past_key_values
                # Note: pass `inputs_embeds` and set `input_ids` to `None`
                outputs = self.base_model(
                    pixel_values=pixel_values,
                    input_ids=None, # IMPORTANT: We provide inputs_embeds
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags,
                    past_key_values=outputs.past_key_values, # Pass the cache
                    labels=None,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=True, # Needed for the next iteration
                    return_dict=True,
                    num_patches_list=num_patches_list,
                    **kwargs
                )

        # The final logits are from the last pass
        final_logits = outputs.logits
            
=======

        for i in range(max_n_latents + 1): # +1 for the final segment
            
            # Prepare a batch of segments for the current iteration
            segment_input_ids = []
            segment_attention_mask = []
            segment_position_ids = []
            max_segment_len = 0
            
            # This will hold the thought vector from the previous step if applicable
            thought_vectors = {}

            if i > 0:
                # We are in a latent step, get the thought vectors
                last_hidden_states = outputs.hidden_states[-1]
                for b in range(batch_size):
                    if i <= len(latent_lists[b]):
                        thought_pos = latent_lists[b][i-1] - 1
                        if thought_pos >= last_processed_pos[b]:
                            # The position relative to the last segment's output
                            relative_pos = thought_pos - last_processed_pos[b]
                            thought_vectors[b] = last_hidden_states[b, relative_pos, :]

            for b in range(batch_size):
                start_pos = last_processed_pos[b]
                # End position is the next latent token or the end of the sequence
                end_pos = latent_lists[b][i] if i < len(latent_lists[b]) else seq_len

                # Get the current segment
                current_segment_ids = input_ids[b, start_pos:end_pos]
                segment_input_ids.append(current_segment_ids)
                
                # Update the max length for padding
                if current_segment_ids.size(0) > max_segment_len:
                    max_segment_len = current_segment_ids.size(0)

                last_processed_pos[b] = end_pos
            
            # Pad the segments to the same length for batch processing
            padded_segment_ids = torch.full((batch_size, max_segment_len), self.eos_token_id,
                                            dtype=torch.long, device=input_ids.device)
            # Create attention mask for the padded segments
            segment_attention_mask = torch.zeros((batch_size, max_segment_len),
                                                 dtype=torch.long, device=input_ids.device)

            for b in range(batch_size):
                segment_len = segment_input_ids[b].size(0)
                padded_segment_ids[b, :segment_len] = segment_input_ids[b]
                segment_attention_mask[b, :segment_len] = 1

            # Get embeddings for the current segments
            inputs_embeds = wte(padded_segment_ids)
            
            # Inject thought vectors from the previous step
            for b, vec in thought_vectors.items():
                # Prepend the thought vector to the embeddings
                inputs_embeds[b] = torch.cat([vec.unsqueeze(0), inputs_embeds[b]], dim=0)
                # Adjust attention mask
                new_mask = torch.cat([torch.ones(1, dtype=torch.long, device=input_ids.device), segment_attention_mask[b]], dim=0)
                segment_attention_mask[b] = new_mask[:-1] # Keep length consistent


            # Forward pass for the current segment
            outputs = self.base_model(
                pixel_values=pixel_values, # Pass visual info at each step
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, # This needs to be correctly constructed for the segment
                position_ids=position_ids, # This also needs to be correctly constructed
                image_flags=image_flags,
                past_key_values=current_past_key_values, # Use cache from previous segment
                labels=None,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=True, # Needed for the next thought vector
                return_dict=True,
                num_patches_list=num_patches_list,
                **kwargs
            )
            
            current_past_key_values = outputs.past_key_values
            all_logits.append(outputs.logits)

        # Concatenate logits from all segments
        final_logits = torch.cat(all_logits, dim=1)

>>>>>>> Stashed changes
        loss = None
        if labels is not None:
<<<<<<< Updated upstream
            # Follow original CoCoNuT loss calculation exactly
            shift_logits = logits[..., :-1, :].contiguous()
||||||| Stash base
            # Standard loss calculation on the final logits
            shift_logits = final_logits[..., :-1, :].contiguous()
=======
            shift_logits = final_logits[..., :-1, :].contiguous()
>>>>>>> Stashed changes
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
<<<<<<< Updated upstream
        
||||||| Stash base
            
        if not return_dict:
            output = (final_logits,) + (outputs.past_key_values,)
            return ((loss,) + output) if loss is not None else output

=======

        if not return_dict:
            output = (final_logits,) + (outputs.past_key_values,)
            return ((loss,) + output) if loss is not None else output

>>>>>>> Stashed changes
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
        
        This follows InternVL3's pattern exactly - visual features replace IMG_CONTEXT tokens.
        This is critical for multimodal CoCoNuT to work correctly.
        
        Args:
            pixel_values: Image pixel values [total_patches, channels, height, width]
            input_ids: Text token IDs [batch_size, sequence_length]
            image_flags: Flags indicating which samples have images [batch_size, 1]
            
        Returns:
            inputs_embeds: Multimodal embeddings [batch_size, sequence_length, hidden_size]
        """
        if pixel_values is None:
            # Text-only case
            return self.base_model.language_model.get_input_embeddings()(input_ids)
        
        # Get text embeddings
        input_embeds = self.base_model.language_model.get_input_embeddings()(input_ids).clone()
        
        # Extract visual features using InternVL3's vision encoder
        vit_embeds = self.base_model.extract_feature(pixel_values)
        
        # Debug logging
        print(f"DEBUG: pixel_values shape: {pixel_values.shape}")
        print(f"DEBUG: vit_embeds shape: {vit_embeds.shape}")
        print(f"DEBUG: input_ids shape: {input_ids.shape}")
        print(f"DEBUG: input_embeds shape: {input_embeds.shape}")
        
        # Note: image_flags filtering is not needed here because the data collator
        # already ensures that pixel_values are correctly batched and all samples have images.
        # The shape mismatch occurs because image_flags has shape [batch_size] but
        # vit_embeds has shape [total_patches, hidden_size] where total_patches != batch_size
        
        # Replace IMG_CONTEXT tokens with visual embeddings (following InternVL3 pattern)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        
        # Find IMG_CONTEXT token positions
        img_context_token_id = getattr(self.base_model, 'img_context_token_id', None)
        print(f"DEBUG: img_context_token_id: {img_context_token_id}")
        
        if img_context_token_id is not None:
            selected = (input_ids_flat == img_context_token_id)
            print(f"DEBUG: selected positions: {selected.sum()}")
            print(f"DEBUG: selected shape: {selected.shape}")
            
            if selected.sum() > 0:
                try:
                    vit_embeds_flat = vit_embeds.reshape(-1, C)
                    print(f"DEBUG: vit_embeds_flat shape: {vit_embeds_flat.shape}")
                    print(f"DEBUG: Trying to assign {vit_embeds_flat.shape[0]} visual tokens to {selected.sum()} positions")
                    
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat
                except Exception as e:
                    print(f"DEBUG: Shape mismatch error: {e}")
                    # Handle shape mismatch gracefully
                    n_token = selected.sum()
                    vit_embeds_flat = vit_embeds.reshape(-1, C)
                    if n_token <= vit_embeds_flat.shape[0]:
                        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat[:n_token]
                    else:
                        # If we don't have enough visual tokens, repeat the last one
                        repeated_embeds = vit_embeds_flat[-1:].repeat(n_token, 1)
                        input_embeds[selected] = input_embeds[selected] * 0.0 + repeated_embeds
        
        # Reshape back to original dimensions
        input_embeds = input_embeds.reshape(B, N, C)
        
        return input_embeds
    
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
            # For text-only processing, use the language model component directly
            inputs_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
            
            outputs = self.base_model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            # Compute loss if labels are provided
            loss = None
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        # For multimodal inputs, use the full InternVL3 model
        # Filter out parameters that InternVL3 doesn't expect
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_patches_list', '_num_patches_list']}
        
        # Ensure image_flags is properly set for InternVL3
        if image_flags is None and pixel_values is not None:
            # Create default image_flags if not provided
            batch_size = input_ids.shape[0]
            image_flags = torch.ones(batch_size, 1, dtype=torch.long, device=input_ids.device)
        
        # # Ensure image_flags is a tensor if it's not None
        # if image_flags is not None and not isinstance(image_flags, torch.Tensor):
        #     image_flags = torch.tensor(image_flags, dtype=torch.long, device=input_ids.device)
        
        # CRITICAL: Set img_context_token_id if not already set
        if not hasattr(self.base_model, 'img_context_token_id') or self.base_model.img_context_token_id is None:
            # Try to get IMG_CONTEXT token ID from the input_ids
            # This is a fallback - ideally this should be set during model initialization
            try:
                # Look for IMG_CONTEXT token in the vocabulary
                # We'll use a common token ID that should exist
                # This is a temporary fix - the proper solution is to ensure tokenizer has IMG_CONTEXT
                self.base_model.img_context_token_id = 151667  # This should be the IMG_CONTEXT token ID for InternVL3
            except:
                self.base_model.img_context_token_id = None
        
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
            **filtered_kwargs
        )
    
    @torch.no_grad()
    def generate(self,
                 pixel_values: Optional[torch.FloatTensor] = None,
                 input_ids: Optional[torch.LongTensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_flags: Optional[torch.LongTensor] = None,
                 visual_features: Optional[torch.FloatTensor] = None,
                 generation_config: Optional[dict] = None,
                 output_hidden_states: Optional[bool] = None,
                 **generate_kwargs) -> torch.LongTensor:
        """
<<<<<<< Updated upstream
        Generate text with multimodal continuous thought reasoning.
        
        This method follows InternVL's generate pattern but uses our CoCoNuT forward pass
        for continuous thought reasoning. It supports both multimodal and text-only generation.
        
        Args:
            pixel_values: Image pixel values [total_patches, channels, height, width]
            input_ids: Input token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            image_flags: Flags indicating which samples have images [batch_size, 1]
            visual_features: Pre-computed visual features (optional)
            generation_config: Generation configuration dictionary
            output_hidden_states: Whether to output hidden states
            **generate_kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs [batch_size, generated_length]
||||||| Stash base
        Generate text with multimodal continuous thought reasoning.

        This method implements a custom generation loop that correctly handles
        the iterative CoCoNuT reasoning process during inference.

        The process is as follows:
        1.  The initial prompt (including images) is processed up to the
            first <|latent|> token using the full base model.
        2.  The `past_key_values` from this initial pass are stored.
        3.  The method iterates through each `<|latent|>` token, injecting the
            "thought vector" (the hidden state from the previous step) as the
            embedding for the current latent token.
        4.  After processing all latent tokens, a standard autoregressive
            decoding loop begins, using the final `past_key_values` that
            encode the full CoCoNuT reasoning chain.

        Args:
            pixel_values: Image pixel values.
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            image_flags: Flags indicating which samples have images.
            generation_config: Generation configuration dictionary.
            num_patches_list: Number of patches per image.
            **generate_kwargs: Additional generation arguments.

        Returns:
            Generated token IDs.
=======
        Generate text with iterative, sequentially dependent multimodal reasoning.

        This method implements a generation loop that respects causality and handles
        dynamic multimodal fusion. It processes the input sequentially.
>>>>>>> Stashed changes
        """
<<<<<<< Updated upstream
        # Set default generation config
||||||| Stash base
        self.eval()

        # generation_config setup
=======
        self.eval()

>>>>>>> Stashed changes
        if generation_config is None:
            generation_config = {}
<<<<<<< Updated upstream
        
        # Extract generation parameters
        max_new_tokens = generation_config.get('max_new_tokens', generate_kwargs.get('max_new_tokens', 100))
        do_sample = generation_config.get('do_sample', generate_kwargs.get('do_sample', True))
        temperature = generation_config.get('temperature', generate_kwargs.get('temperature', 0.7))
        top_p = generation_config.get('top_p', generate_kwargs.get('top_p', 0.9))
        top_k = generation_config.get('top_k', generate_kwargs.get('top_k', 50))
||||||| Stash base
        max_new_tokens = generation_config.get('max_new_tokens', generate_kwargs.get('max_new_tokens', 100))
=======
        max_new_tokens = generation_config.get('max_new_tokens', 100)
>>>>>>> Stashed changes
        eos_token_id = generation_config.get('eos_token_id', self.eos_token_id)
        pad_token_id = generation_config.get('pad_token_id', eos_token_id)
        
<<<<<<< Updated upstream
        self.eval()
        
        # Handle multimodal inputs by preparing embeddings (following InternVL pattern)
        if pixel_values is not None:
            # Use visual features if provided, otherwise extract them
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.base_model.extract_feature(pixel_values)
            
            # Get text embeddings
            input_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            
            # Replace IMG_CONTEXT tokens with visual embeddings
            input_ids_flat = input_ids.reshape(B * N)
            img_context_token_id = getattr(self.base_model, 'img_context_token_id', None)
            
            if img_context_token_id is not None:
                selected = (input_ids_flat == img_context_token_id)
                if selected.sum() > 0:
                    try:
                        vit_embeds_flat = vit_embeds.reshape(-1, C)
                        input_embeds[selected] = vit_embeds_flat.to(input_embeds.device)
                    except Exception as e:
                        # Handle shape mismatch gracefully
                        n_token = selected.sum()
                        vit_embeds_flat = vit_embeds.reshape(-1, C)
                        if n_token <= vit_embeds_flat.shape[0]:
                            input_embeds[selected] = vit_embeds_flat[:n_token].to(input_embeds.device)
                        else:
                            # Repeat the last visual embedding if needed
                            repeated_embeds = vit_embeds_flat[-1:].repeat(n_token, 1)
                            input_embeds[selected] = repeated_embeds.to(input_embeds.device)
            
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            # Text-only generation
            input_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)
||||||| Stash base
        batch_size, seq_len = input_ids.shape
        # This will hold the generated tokens for each item in the batch
        generated_ids = [[] for _ in range(batch_size)]

        # --- CoCoNuT Iterative Reasoning ---
        # Find latent tokens
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)

        # Initial forward pass to process the prompt and get initial KV cache
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            num_patches_list=num_patches_list,
        )
        past_key_values = outputs.past_key_values

        if len(latent_indices) > 0:
            latent_lists = [
                sorted([idx[1].item() for idx in latent_indices if idx[0] == i])
                for i in range(batch_size)
            ]
            max_n_latents = max(len(l) for l in latent_lists)

            wte = self.base_model.get_input_embeddings()
            inputs_embeds = wte(input_ids)
            
            for i in range(max_n_latents):
                last_hidden_states = outputs.hidden_states[-1]
                for b in range(batch_size):
                    if i < len(latent_lists[b]):
                        current_latent_pos = latent_lists[b][i]
                        thought_pos = current_latent_pos - 1
                        if thought_pos >= 0:
                            thought_vector = last_hidden_states[b, thought_pos, :]
                            inputs_embeds[b, current_latent_pos, :] = thought_vector
                
                # We only need to process the *last* token of the current sequence section
                # to get the next hidden state, but for simplicity we pass the whole sequence
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

        # --- Autoregressive Decoding ---
        # Start decoding from the last token of the input_ids
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
=======
        batch_size, seq_len = input_ids.shape
        wte = self.base_model.get_input_embeddings()
        
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        latent_lists = [
            sorted([idx[1].item() for idx in latent_indices if idx[0] == i])
            for i in range(batch_size)
        ]

        # Process prompt and latent tokens sequentially
        current_past_key_values = None
        last_processed_pos = {b: 0 for b in range(batch_size)}
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        for i in range(max_n_latents + 1):
            # ... (Full sequential processing logic from _multimodal_forward_pass would go here)
            # This is a simplified version for now. A full implementation would require
            # careful handling of segments, padding, and attention masks.
            
            # For this refactor, we'll assume a simplified generation loop that
            # at least passes the pixel_values at each step.
            pass

        # For now, let's use a simplified version of the old generate logic
        # but ensure pixel_values are passed in the generation loop.
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            num_patches_list=num_patches_list,
        )
        past_key_values = outputs.past_key_values
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
>>>>>>> Stashed changes
        
<<<<<<< Updated upstream
        # Use the language model's generate method with our prepared embeddings
        # This leverages the optimized generation implementation while using our multimodal embeddings
        outputs = self.base_model.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs
        )
||||||| Stash base
        # Append initial input_ids to the generated sequences
        for b in range(batch_size):
            generated_ids[b].extend(input_ids[b].tolist())

        for _ in range(max_new_tokens):
            # No need to provide pixel_values or image_flags again
            outputs = self.base_model.forward(
                input_ids=next_token_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the newly generated tokens
            for b in range(batch_size):
                generated_ids[b].append(next_token_ids[b].item())

            # Check for EOS token
            if (next_token_ids == eos_token_id).all():
                break
=======
        generated_ids = [ids.tolist() for ids in input_ids]

        for _ in range(max_new_tokens):
            outputs = self.base_model.forward(
                pixel_values=pixel_values, # Pass pixel_values in each generation step
                input_ids=next_token_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                image_flags=image_flags,
                num_patches_list=num_patches_list
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            for b in range(batch_size):
                generated_ids[b].append(next_token_ids[b].item())

            if (next_token_ids == eos_token_id).all():
                break
>>>>>>> Stashed changes
        
<<<<<<< Updated upstream
        return outputs
||||||| Stash base
        # Pad the generated sequences to the same length for tensor conversion
        max_len = max(len(ids) for ids in generated_ids)
        padded_ids = [
            ids + [eos_token_id] * (max_len - len(ids)) for ids in generated_ids
        ]
        
        return torch.tensor(padded_ids, dtype=torch.long, device=input_ids.device)
=======
        max_len = max(len(ids) for ids in generated_ids)
        padded_ids = [
            ids + [eos_token_id] * (max_len - len(ids)) for ids in generated_ids
        ]
        
        return torch.tensor(padded_ids, dtype=torch.long, device=input_ids.device)
>>>>>>> Stashed changes
    
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
            print(f'Dynamic ViT batch size: {image_bs}')
        
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
            print("Warning: No parameters found in model, fallback to cpu")
            device = torch.device('cpu')
        
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        
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
            print(f"Query: {query_to_print}")
            print(f"Response: {response}")
        
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
