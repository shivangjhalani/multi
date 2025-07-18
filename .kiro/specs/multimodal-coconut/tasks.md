# Implementation Plan

## Core Architecture Principles - NEVER FORGET

You have access to coconut codebase and InternVL codebase in folder:reference
#[[file:reference/coconut/coconut.md]]

**The essence of CoCoNuT that must be preserved throughout implementation:**

1. **Continuous Thought Mechanism**: The core innovation is replacing discrete textual reasoning steps with continuous vector representations (hidden states). When encountering `<|latent|>` tokens, we feed the previous hidden state as the input embedding for the current position.

2. **Staged Curriculum Learning**: Training progresses through stages where we gradually replace more textual reasoning steps with latent tokens. Stage 0 = full CoT, Stage k = first k reasoning steps replaced with latent tokens.

3. **Iterative Forward Passes**: Cannot do single forward pass due to dependency - latent token at position i depends on hidden state at position i-1. Must process sequence in chunks with KV cache for efficiency.

4. **Multi-Pass Architecture**: 
   - Find first latent token position
   - Forward pass up to that position  
   - Use resulting hidden state as embedding for latent token
   - Repeat for each subsequent latent token
   - Final forward pass for remaining sequence

5. **KV Cache Optimization**: Store and reuse Key/Value matrices from previous passes to avoid recomputation. Critical for making multi-pass approach feasible.

6. **Multimodal Extension**: All above principles apply but now hidden states encode both visual and textual information from InternVL3's multimodal representations.

**Key Implementation Checkpoints:**
- [ ] Verify latent token detection works correctly in multimodal context
- [ ] Ensure hidden state feedback loop preserves multimodal information  
- [ ] Confirm KV cache handles both visual and text tokens efficiently
- [ ] Validate staged curriculum progresses correctly with image+text data
- [ ] Test that continuous thoughts improve reasoning over discrete steps

**Reference Implementation Pattern (from original CoCoNuT):**
```python
# Core pattern that MUST be preserved in multimodal version
latent_indices = (input_ids == self.latent_token_id).nonzero()
for pass_idx in range(max_n_latents):
    # Forward pass up to next latent token
    outputs = self.base_model(inputs_embeds=..., past_key_values=kv_cache)
    # Extract hidden state and use as embedding for latent token
    hidden_state = outputs.hidden_states[-1][:, -1, :]
    inputs_embeds[:, latent_pos, :] = hidden_state
```

## General Implementation Guidelines

**Development Philosophy:**
- **Keep It Simple**: Follow the original CoCoNuT's elegant simplicity. Don't over-architect or add unnecessary complexity.
- **Incremental Development**: Build and test each component thoroughly before moving to the next.
- **Code Reuse**: Leverage existing CoCoNuT patterns and InternVL3 components wherever possible.
- **Minimal Changes**: Make the smallest possible changes to achieve multimodal functionality.

**Research and Documentation:**
- **Use Context7 MCP Server**: Always use the Context7 MCP server to get up-to-date documentation for libraries and frameworks:
  ```bash
  # Example: Get InternVL documentation
  mcp_context7_get_library_docs --library="/opengvlab/internvl" --topic="model architecture"
  ```
- **Reference Original Code**: Constantly refer back to `reference/coconut/` for implementation patterns.
- **Document Decisions**: Keep implementation notes for complex design choices.

**Code Quality Standards:**
- **Follow Original Style**: Match the coding style and patterns from the original CoCoNuT codebase.
- **Minimal Dependencies**: Only add dependencies that are absolutely necessary.
- **Error Handling**: Implement robust error handling but keep it simple and informative.
- **Testing First**: Write tests as you implement, not after.

**Performance Considerations:**
- **Memory Efficiency**: Always consider memory usage, especially for multimodal data.
- **GPU Utilization**: Optimize for efficient GPU usage and batch processing.
- **Avoid Premature Optimization**: Focus on correctness first, then optimize bottlenecks.

**Integration Guidelines:**
- **Focus on Multimodal**: Build purely for multimodal functionality without text-only legacy support.
- **Configuration Consistency**: Follow the original YAML configuration patterns.
- **Distributed Training**: Design with multi-GPU training in mind from the start.

**Debugging and Validation:**
- **Sanity Checks**: Add assertions and validation at critical points.
- **Logging**: Use informative logging but avoid spam.
- **Visualization**: Create simple tools to visualize multimodal data and model outputs.

**Common Pitfalls to Avoid:**
- **Over-engineering**: Don't create complex abstractions unless absolutely needed.
- **Ignoring Edge Cases**: Handle variable image sizes, sequence lengths, and batch compositions.
- **Memory Leaks**: Be careful with tensor operations and GPU memory management.
- **Breaking CoCoNuT Logic**: Always validate that core continuous thought mechanism works correctly.

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure for multimodal CoCoNuT implementation
  - Set up configuration system with YAML support and validation
  - Implement utility functions for distributed training and logging
  - Create requirements.txt with all necessary dependencies
  - _Requirements: 5.1, 5.2, 5.3, 6.3_

- [x] 2. Implement multimodal data pipeline foundation
  - [x] 2.1 Create base multimodal dataset class
    - Implement MultimodalDataset class that extends original CoCoNuT dataset structure
    - Add image loading and validation functionality
    - Implement tokenization for multimodal samples with special token integration
    - _Requirements: 1.1, 1.2, 1.4, 6.4_

  - [x] 2.2 Implement image preprocessing pipeline
    - Integrate InternVL3's image preprocessing functions (dynamic_preprocess, build_transform)
    - Add image resizing, normalization, and tiling functionality
    - Implement error handling for corrupted or missing images
    - _Requirements: 1.3, 1.1_

  - [x] 2.3 Create multimodal data collator
    - Implement MultimodalCollator class for efficient batching
    - Add logic to align latent tokens across batch samples
    - Handle variable image patch counts and sequence lengths
    - _Requirements: 1.5, 6.5_

  - [x] 2.4 Prepare A-OKVQA dataset for training and evaluation
    - Create A-OKVQA dataset download and preprocessing scripts with automatic data fetching
    - Convert A-OKVQA format to multimodal CoCoNuT compatible JSON structure (image_path, question, steps, answer).
    - Add support for different A-OKVQA splits (train, val, test) and rationale types (multiple-choice, direct-answer)
    - _Requirements: 1.1, 1.2, 1.6, 4.1, 6.6_

- [ ] 3. Develop core multimodal CoCoNuT model architecture
  - [x] 3.1 Create InternVL3 integrat ion layer
    - Load and initialize InternVL3-1B-Pretrained model
    - Integrate CoCoNuT special tokens into InternVL3's vocabulary
    - Implement model weight loading and checkpoint management
    - _Requirements: 2.1, 2.2, 6.2_

  - [x] 3.2 Implement multimodal forward pass logic
    - Extend original CoCoNuT forward method to handle pixel_values input
    - Implement iterative forward passes with multimodal KV cache
    - Add latent token detection and continuous thought feedback for multimodal inputs
    - _Requirements: 2.3, 2.4, 2.5_

  - [x] 3.3 Create multimodal generation capabilities
    - Implement generate method for multimodal inference
    - Add support for image-conditioned text generation with continuous thoughts
    - Handle mixed text-only and multimodal generation scenarios
    - Create both user-friendly chat() method (for conversational interface) and comprehensive generate() method (for flexible generation)
    - _Requirements: 2.6, 6.5_

- [-] 4. Implement staged training curriculum system
  - [x] 4.1 Create stage management system
    - Implement StageManager class for curriculum progression
    - Add logic to calculate current stage based on epoch and epochs_per_stage
    - Create stage-specific data preparation functions
    - _Requirements: 3.2, 3.3, 3.4_

  - [x] 4.2 Implement multimodal CoT pre-training (Stage 0)
    - Create training loop for standard multimodal chain-of-thought
    - Implement loss calculation for multimodal reasoning steps
    - Add validation logic for CoT pre-training stage
    - _Requirements: 3.1, 3.7_

  - [x] 4.3 Implement progressive latent stage training
    - Create data preparation for replacing reasoning steps with latent tokens
    - Implement training logic for stages 1 through max_latent_stage
    - Add uniform probability mixing for multi-stage data
    - _Requirements: 3.4, 3.5, 3.6_

- [x] 5. Create evaluation and benchmarking system
  - [x] 5.1 Implement A-OKVQA evaluation pipeline
    - Create evaluation dataset loader for A-OKVQA format
    - Implement VQA accuracy metrics calculation
    - Add support for multiple-choice and open-ended question evaluation
    - _Requirements: 4.1, 4.2_

  - [x] 5.2 Create reasoning quality analysis tools
    - Implement tools to inspect continuous thought representations
    - Add visualization for latent space reasoning progression
    - Create comparison metrics between discrete and continuous reasoning
    - _Requirements: 4.3, 4.2_

  - [x] 5.3 Implement efficiency benchmarking
    - Add memory usage profiling during training and inference
    - Implement inference time measurement and comparison
    - Create throughput benchmarks for different batch sizes and configurations
    - _Requirements: 4.4, 4.5_

- [x] 6. Add distributed training and optimization support
  - [x] 6.1 Implement distributed training setup
    - Add FSDP (Fully Sharded Data Parallel) support for large model training
    - Implement DDP (Distributed Data Parallel) as fallback option
    - Add proper synchronization for multimodal batches across processes
    - _Requirements: 5.5, 6.1_

  - [x] 6.2 Create checkpoint management system
    - Implement model state saving and loading for multimodal CoCoNuT
    - Add training state preservation including stage progression
    - Create checkpoint validation and recovery mechanisms
    - _Requirements: 5.6, 6.1_

  - [x] 6.3 Add memory optimization features
    - Implement gradient checkpointing for memory efficiency
    - Add automatic batch size reduction on OOM errors
    - Create KV cache optimization for multimodal sequences
    - _Requirements: 6.1, 6.2_

- [ ] 7. Create configuration and experiment management
  - [x] 7.1 Implement YAML configuration system
    - Create Config class with dynamic attribute setting
    - Add configuration validation and error handling
    - Implement environment variable substitution in configs
    - _Requirements: 5.1, 5.2_

  - [x] 7.2 Create experiment configuration templates
    - Create YAML templates for different training scenarios (CoT, CoCoNuT, evaluation)
    - Add configuration inheritance and composition features
    - Implement runtime configuration updates for stage transitions
    - _Requirements: 5.3, 5.4_

  - [x] 7.3 Add experiment tracking and logging
    - Integrate wandb
    - Add comprehensive logging for training metrics and model performance
    - Create debugging utilities for multimodal training issues
    - _Requirements: 5.4, 5.5_

- [-] 8. Implement comprehensive testing framework
  - [x] 8.1 Create unit tests for core components
    - Write tests for multimodal data pipeline components
    - Add tests for model forward pass and generation logic
    - Create tests for configuration system and utilities
    - _Requirements: 6.1, 6.2, 6.3_

  - [-] 8.2 Implement integration tests
    - Create end-to-end training tests on small datasets
    - Add model compatibility tests with different InternVL3 variants
    - Implement distributed training integration tests
    - _Requirements: 6.4, 6.5, 6.6_

  - [ ] 8.3 Add performance and validation tests
    - Create memory usage and training speed benchmarks
    - Implement reasoning quality validation tests
    - Add robustness tests for various image types and qualities
    - _Requirements: 6.1, 6.2, 6.4_

- [ ] 9. Create documentation and examples
  - [ ] 9.1 Write comprehensive documentation
    - Create README with setup and usage instructions
    - Write API documentation for all major classes and functions
    - Add troubleshooting guide for common issues
    - _Requirements: 6.3, 6.4_

  - [ ] 9.2 Implement example scripts and tutorials
    - Create example training scripts for different scenarios
    - Add inference examples with sample images and questions
    - Write tutorial notebooks demonstrating key features
    - _Requirements: 6.1, 6.2_



- [ ] 10. Final integration and optimization
  - [ ] 10.1 Conduct comprehensive benchmarking
    - Run full training experiments on A-OKVQA dataset
    - Compare performance with baseline InternVL3 model
    - Analyze reasoning quality improvements from continuous thoughts
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 10.2 Optimize performance and memory usage
    - Profile and optimize critical code paths
    - Implement advanced batching strategies for mixed sequence lengths
    - Add support for gradient accumulation and mixed precision training
    - _Requirements: 4.4, 6.1, 6.2_

  - [ ] 10.3 Prepare for production deployment
    - Create deployment scripts and Docker containers
    - Add model serving capabilities for inference
    - Implement model quantization and optimization for deployment
    - _Requirements: 6.1, 6.2, 6.3_