# Requirements Document

## Introduction

This project aims to extend the CoCoNuT (Chain of Continuous Thought) methodology from text-only reasoning to multimodal reasoning that combines images and text. The goal is to enable Large Language Models to reason in continuous latent space while processing both visual and textual information, using InternVL3-1B-Pretrained as the base model and A-OKVQA as the primary dataset.

CoCoNuT represents a paradigm shift from discrete textual reasoning steps to continuous thought vectors, allowing models to reason in a high-dimensional latent space. This project will adapt this approach to handle multimodal inputs, creating a system that can perform visual question answering through continuous latent reasoning.

## Requirements

### Requirement 1: Multimodal Data Pipeline

**User Story:** As a researcher, I want to process multimodal datasets (image + text) in a format compatible with CoCoNuT training, so that I can train models to reason over both visual and textual information.

#### Acceptance Criteria

1. WHEN processing A-OKVQA dataset THEN the system SHALL load and preprocess both images and question-answer pairs
2. WHEN formatting training data THEN the system SHALL create (image, question, reasoning_steps, answer) tuples compatible with CoCoNuT curriculum
3. WHEN handling images THEN the system SHALL resize and normalize images according to InternVL3 preprocessing requirements
4. WHEN tokenizing text THEN the system SHALL use InternVL3's tokenizer while preserving CoCoNuT special tokens (<|latent|>, <|start-latent|>, <|end-latent|>)
5. WHEN creating batches THEN the system SHALL efficiently batch both image tensors and tokenized text sequences
6. IF reasoning steps are not available in the dataset THEN the system SHALL generate intermediate reasoning steps using a teacher model or heuristic approach

### Requirement 2: Multimodal CoCoNuT Architecture

**User Story:** As a researcher, I want to integrate InternVL3's vision-language capabilities with CoCoNuT's continuous reasoning mechanism, so that the model can perform latent space reasoning over multimodal inputs.

#### Acceptance Criteria

1. WHEN initializing the model THEN the system SHALL load InternVL3-1B-Pretrained weights as the base multimodal model
2. WHEN processing inputs THEN the system SHALL encode images using InternVL3's vision encoder and combine with text embeddings
3. WHEN encountering <|latent|> tokens THEN the system SHALL perform continuous thought feedback using multimodal hidden states
4. WHEN performing forward passes THEN the system SHALL maintain compatibility with InternVL3's attention mechanisms while implementing CoCoNuT's iterative processing
5. WHEN caching key-value pairs THEN the system SHALL efficiently handle both visual and textual tokens in the KV cache
6. WHEN generating responses THEN the system SHALL produce coherent answers based on both visual and textual reasoning

### Requirement 3: Staged Training Curriculum for Multimodal Reasoning

**User Story:** As a researcher, I want to implement a staged training curriculum that gradually transitions from explicit multimodal reasoning steps to continuous latent reasoning, so that the model learns to reason effectively in the continuous space.

#### Acceptance Criteria

1. WHEN starting training THEN the system SHALL begin with standard multimodal chain-of-thought (Stage 0)
2. WHEN progressing through stages THEN the system SHALL gradually replace textual reasoning steps with <|latent|> tokens
3. WHEN calculating stage progression THEN the system SHALL use epoch-based scheduling (scheduled_stage = epoch // epochs_per_stage)
4. WHEN preparing stage-specific data THEN the system SHALL replace the first k reasoning steps with k * c_thought latent tokens
5. WHEN training at each stage THEN the system SHALL only supervise the remaining textual steps and final answer
6. WHEN reaching maximum latent stage THEN the system SHALL perform most reasoning in continuous space with minimal textual output
7. IF uniform_prob is specified THEN the system SHALL mix data from different stages during training

### Requirement 4: Evaluation and Benchmarking System

**User Story:** As a researcher, I want to evaluate the multimodal CoCoNuT model's performance on visual question answering tasks, so that I can measure the effectiveness of continuous latent reasoning for multimodal problems.

#### Acceptance Criteria

1. WHEN evaluating on A-OKVQA THEN the system SHALL report accuracy metrics comparable to standard VQA evaluation
2. WHEN comparing performance THEN the system SHALL benchmark against baseline InternVL3 model without CoCoNuT modifications
3. WHEN analyzing reasoning quality THEN the system SHALL provide mechanisms to inspect the continuous thought process
4. WHEN measuring efficiency THEN the system SHALL track inference time and memory usage compared to discrete reasoning approaches
5. WHEN validating during training THEN the system SHALL perform periodic evaluation to monitor learning progress
6. IF multiple datasets are used THEN the system SHALL report performance on each dataset separately

### Requirement 5: Configuration and Experiment Management

**User Story:** As a researcher, I want flexible configuration options for multimodal CoCoNuT experiments, so that I can easily adjust hyperparameters and experiment settings.

#### Acceptance Criteria

1. WHEN configuring experiments THEN the system SHALL provide YAML configuration files for different training stages
2. WHEN setting hyperparameters THEN the system SHALL allow adjustment of c_thought, epochs_per_stage, max_latent_stage, and multimodal-specific parameters
3. WHEN specifying model paths THEN the system SHALL support loading pre-trained InternVL3 checkpoints and CoCoNuT model states
4. WHEN managing datasets THEN the system SHALL allow configuration of dataset paths, preprocessing options, and batch sizes
5. WHEN running distributed training THEN the system SHALL support multi-GPU training with proper synchronization
6. WHEN saving checkpoints THEN the system SHALL preserve both model weights and training state for resumable training

### Requirement 6: Integration and Extensibility

**User Story:** As a researcher, I want the multimodal CoCoNuT implementation to leverage InternVL3's architecture effectively while following CoCoNuT principles, so that I can build a robust multimodal reasoning system.

#### Acceptance Criteria

1. WHEN using InternVL3 components THEN the system SHALL preserve the model's original multimodal capabilities
2. WHEN implementing new features THEN the system SHALL follow the existing CoCoNuT code structure and patterns
3. WHEN handling special tokens THEN the system SHALL properly integrate CoCoNuT tokens with InternVL3's vocabulary
4. WHEN processing multimodal batches THEN the system SHALL handle variable image sizes and sequence lengths efficiently
5. WHEN extending to other datasets THEN the system SHALL provide a flexible framework for additional multimodal datasets
6. WHEN implementing the architecture THEN the system SHALL focus purely on multimodal functionality without text-only legacy support