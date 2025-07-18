#!/usr/bin/env python3
"""
Multimodal CoCoNuT Tutorial Notebook

This comprehensive tutorial demonstrates the key features and capabilities
of the Multimodal CoCoNuT system. It's designed to be run step-by-step
to understand the core concepts and implementation.

This can be converted to a Jupyter notebook or run as a Python script.

Usage:
    python examples/tutorial_notebook.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def tutorial_section(title: str):
    """Decorator to mark tutorial sections"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print("\n" + "="*80)
            print(f"TUTORIAL SECTION: {title}")
            print("="*80)
            return func(*args, **kwargs)
        return wrapper
    return decorator


@tutorial_section("1. Introduction to Multimodal CoCoNuT")
def section_1_introduction():
    """
    Introduction to the core concepts of Multimodal CoCoNuT
    """
    print("""
    Welcome to the Multimodal CoCoNuT Tutorial!
    
    CoCoNuT (Chain of Continuous Thought) represents a paradigm shift from discrete 
    textual reasoning steps to continuous thought vectors. This multimodal extension 
    adapts this approach to handle image-text pairs for visual question answering.
    
    Key Concepts:
    
    1. CONTINUOUS THOUGHTS: Instead of generating explicit reasoning steps as text,
       the model uses continuous vector representations in a high-dimensional latent space.
    
    2. LATENT TOKENS: Special <|latent|> tokens mark positions where continuous 
       thoughts should be inserted during training and inference.
    
    3. STAGED CURRICULUM: Training progresses through stages:
       - Stage 0: Standard multimodal chain-of-thought
       - Stage k: First k reasoning steps replaced with latent tokens
       - Progressive deepening of continuous reasoning
    
    4. MULTIMODAL INTEGRATION: Built on InternVL3-1B-Pretrained for vision-language 
       understanding, combining visual and textual reasoning.
    
    5. ITERATIVE PROCESSING: Multiple forward passes with KV cache optimization
       handle the dependency chain between continuous thoughts.
    """)
    
    # Show the core algorithm conceptually
    print("\nCore Algorithm Overview:")
    print("1. Detect latent token positions in input sequence")
    print("2. Perform iterative forward passes:")
    print("   a. Process tokens up to next latent position")
    print("   b. Extract hidden states from previous position")
    print("   c. Replace latent token with continuous thought (hidden state)")
    print("   d. Continue with updated embeddings")
    print("3. Use KV cache for efficiency during multi-pass processing")
    print("4. Concatenate logits and compute final loss")


@tutorial_section("2. Configuration System")
def section_2_configuration():
    """
    Demonstrate the configuration system
    """
    from multimodal_coconut import (
        Config, 
        create_config_from_template, 
        validate_config,
        print_config_summary
    )
    
    print("The configuration system is the foundation of Multimodal CoCoNuT.")
    print("It provides flexible, validated, and template-based configuration management.\n")
    
    # Create different configurations
    print("Creating different configuration templates:")
    
    # CoT configuration
    cot_config = create_config_from_template('cot')
    print("✓ CoT (Chain-of-Thought) pre-training configuration")
    
    # CoCoNuT configuration  
    coconut_config = create_config_from_template('coconut')
    print("✓ CoCoNuT training configuration")
    
    # Debug configuration
    debug_config = create_config_from_template('debug')
    print("✓ Debug configuration (small scale for testing)")
    
    # Custom configuration
    custom_config = create_config_from_template(
        'coconut',
        name="tutorial-experiment",
        c_thought=3,
        max_latent_stage=6,
        batch_size_training=4,
        learning_rate=5e-6
    )
    print("✓ Custom configuration with overrides")
    
    # Show configuration summary
    print("\nCustom Configuration Summary:")
    print_config_summary(custom_config)
    
    # Demonstrate validation
    print("\nConfiguration Validation:")
    try:
        validate_config(custom_config)
        print("✓ Configuration is valid")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
    
    return custom_config


@tutorial_section("3. Model Architecture")
def section_3_model_architecture():
    """
    Explore the MultimodalCoconut model architecture
    """
    from multimodal_coconut import MultimodalCoconut
    from transformers import AutoTokenizer
    
    print("The MultimodalCoconut model extends InternVL3 with continuous thought capabilities.\n")
    
    # Create a simple mock model for demonstration
    print("Creating model components:")
    
    # Mock base model (in practice, this would be InternVL3)
    class MockInternVL3:
        def __init__(self):
            self.language_model = torch.nn.Linear(64, 1000)  # Mock language model
            self.language_model.config = type('Config', (), {'hidden_size': 64})()
            self.img_context_token_id = 151667
        
        def extract_feature(self, pixel_values):
            # Mock visual feature extraction
            batch_size = pixel_values.shape[0] if pixel_values is not None else 1
            return torch.randn(batch_size * 12, 64)  # 12 patches per image
    
    base_model = MockInternVL3()
    print("✓ Mock base model created (InternVL3)")
    
    # Create tokenizer with special tokens
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    except:
        # Fallback for offline environments
        print("Warning: Using mock tokenizer")
        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.eos_token_id = 50256
                self.pad_token_id = 50256
            def convert_tokens_to_ids(self, token):
                return {"<|latent|>": 50257, "<|start-latent|>": 50258, "<|end-latent|>": 50259}.get(token, 0)
        tokenizer = MockTokenizer()
    
    # Add special tokens
    special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
    if hasattr(tokenizer, 'add_special_tokens'):
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    print("✓ Tokenizer with special tokens created")
    
    # Get special token IDs
    latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos_token_id = tokenizer.eos_token_id
    
    print(f"Special token IDs: latent={latent_token_id}, start={start_latent_id}, end={end_latent_id}")
    
    # Create MultimodalCoconut model
    model = MultimodalCoconut(
        base_model=base_model,
        latent_token_id=latent_token_id,
        start_latent_id=start_latent_id,
        end_latent_id=end_latent_id,
        eos_token_id=eos_token_id
    )
    
    print("✓ MultimodalCoconut model created")
    print(f"Model hidden size: {model.hidden_size}")
    
    # Demonstrate model components
    print("\nModel Architecture Components:")
    print("1. Base Model: InternVL3-1B-Pretrained (vision + language)")
    print("2. Continuous Thought Mechanism: Hidden state feedback")
    print("3. Iterative Processing: Multi-pass forward with KV cache")
    print("4. Special Token Handling: Latent token detection and replacement")
    
    return model, tokenizer


@tutorial_section("4. Data Pipeline")
def section_4_data_pipeline():
    """
    Demonstrate the multimodal data pipeline
    """
    from multimodal_coconut.data import ImageProcessor, MultimodalDataset, MultimodalCollator
    
    print("The data pipeline handles multimodal inputs (images + text) for CoCoNuT training.\n")
    
    # Create image processor
    image_processor = ImageProcessor(
        image_size=448,
        max_num_patches=12,
        use_thumbnail=True,
        dynamic_preprocess=True
    )
    print("✓ Image processor created")
    print(f"  Image size: {image_processor.image_size}")
    print(f"  Max patches: {image_processor.max_num_patches}")
    
    # Create sample data
    sample_data = [
        {
            "question": "What is in the image?",
            "steps": [
                "I need to look at the image carefully.",
                "I can see objects in the scene.",
                "Let me identify the main subject."
            ],
            "answer": "A cat sitting on a chair",
            "image": "sample_image.jpg"
        }
    ]
    
    print("\n✓ Sample data structure:")
    print(json.dumps(sample_data[0], indent=2))
    
    # Demonstrate tokenization process
    print("\nTokenization Process:")
    print("1. Combine question, reasoning steps, and answer")
    print("2. Insert <|latent|> tokens at reasoning positions")
    print("3. Create input_ids and labels for training")
    print("4. Process images into patch embeddings")
    
    # Show how latent tokens are inserted
    sample = sample_data[0]
    reasoning_text = " ".join(sample["steps"])
    full_text = f"Question: {sample['question']}\nReasoning: {reasoning_text}\nAnswer: {sample['answer']}"
    
    print(f"\nOriginal text:\n{full_text}")
    
    # Simulate latent token insertion (Stage 1: first reasoning step becomes latent)
    latent_text = f"Question: {sample['question']}\nReasoning: <|latent|> {' '.join(sample['steps'][1:])}\nAnswer: {sample['answer']}"
    print(f"\nWith latent tokens (Stage 1):\n{latent_text}")
    
    return image_processor


@tutorial_section("5. Training Process")
def section_5_training_process():
    """
    Explain the staged training process
    """
    from multimodal_coconut.training import StageManager
    
    print("CoCoNuT uses a staged curriculum learning approach.\n")
    
    # Create stage manager
    stage_manager = StageManager(
        max_latent_stage=4,
        epochs_per_stage=5,
        c_thought=2
    )
    
    print("Training Stages:")
    print("Stage 0: Standard multimodal chain-of-thought (explicit reasoning)")
    
    for stage in range(1, 5):
        epoch = stage * 5  # Assuming 5 epochs per stage
        current_stage = stage_manager.get_current_stage(epoch)
        print(f"Stage {stage}: First {stage} reasoning steps → latent tokens")
    
    print("\nStage Progression Example:")
    original_steps = [
        "I need to examine the image carefully.",
        "I can see a furry animal with whiskers.",
        "The animal has pointed ears and a long tail.",
        "Based on these features, this is a cat."
    ]
    
    print("Original reasoning steps:")
    for i, step in enumerate(original_steps, 1):
        print(f"  {i}. {step}")
    
    print("\nStage transformations:")
    for stage in range(1, 5):
        transformed = []
        for i, step in enumerate(original_steps):
            if i < stage:
                transformed.append("<|latent|>")
            else:
                transformed.append(step)
        
        print(f"\nStage {stage}:")
        for i, item in enumerate(transformed, 1):
            print(f"  {i}. {item}")
    
    print("\nKey Benefits of Staged Training:")
    print("1. Gradual transition from explicit to implicit reasoning")
    print("2. Model learns to compress reasoning into continuous representations")
    print("3. Maintains reasoning quality while reducing token usage")
    print("4. Enables faster inference with latent thoughts")


@tutorial_section("6. Continuous Thought Mechanism")
def section_6_continuous_thoughts():
    """
    Demonstrate the continuous thought mechanism
    """
    print("The continuous thought mechanism is the core innovation of CoCoNuT.\n")
    
    # Simulate the continuous thought process
    print("Continuous Thought Process:")
    print("1. Model processes tokens up to <|latent|> position")
    print("2. Extracts hidden state from previous position")
    print("3. Replaces <|latent|> token embedding with hidden state")
    print("4. Continues processing with 'continuous thought'")
    
    # Visual representation
    print("\nVisual Representation:")
    print("Input:  [Question] [Image] [<|latent|>] [Answer]")
    print("                              ↓")
    print("        [Question] [Image] → [Hidden State] → [Answer]")
    print("                              ↑")
    print("                    Continuous Thought")
    
    # Demonstrate with mock tensors
    print("\nMock Tensor Example:")
    
    # Simulate input embeddings
    batch_size, seq_len, hidden_size = 2, 10, 64
    input_embeds = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Input embeddings shape: {input_embeds.shape}")
    
    # Simulate latent token positions
    latent_positions = torch.tensor([[0, 5], [1, 7]])  # [batch_idx, token_idx]
    print(f"Latent positions: {latent_positions}")
    
    # Simulate hidden states from previous processing
    hidden_states = torch.randn(batch_size, seq_len-1, hidden_size)
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Simulate continuous thought replacement
    for batch_idx, token_idx in latent_positions:
        if token_idx > 0:  # Ensure we have a previous position
            # Replace latent token with previous hidden state
            continuous_thought = hidden_states[batch_idx, token_idx-1, :]
            input_embeds[batch_idx, token_idx, :] = continuous_thought
            print(f"Replaced latent token at [{batch_idx}, {token_idx}] with continuous thought")
    
    print("\n✓ Continuous thought replacement completed")
    print("The model now processes 'thoughts' instead of discrete reasoning text")


@tutorial_section("7. Inference and Generation")
def section_7_inference():
    """
    Demonstrate inference and generation process
    """
    print("Inference with Multimodal CoCoNuT combines visual understanding with continuous reasoning.\n")
    
    # Mock inference setup
    print("Inference Setup:")
    print("1. Load trained model with continuous thought capabilities")
    print("2. Process input image and question")
    print("3. Generate response using continuous thoughts")
    
    # Simulate inference process
    print("\nInference Process:")
    
    # Mock inputs
    question = "What is the cat doing in the image?"
    print(f"Question: {question}")
    
    # Mock image processing
    print("Processing image...")
    mock_pixel_values = torch.randn(1, 3, 448, 448)  # Mock image tensor
    print(f"Image tensor shape: {mock_pixel_values.shape}")
    
    # Mock tokenization
    print("Tokenizing question...")
    mock_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # Mock token IDs
    print(f"Input IDs shape: {mock_input_ids.shape}")
    
    # Mock generation process
    print("\nGeneration Process:")
    print("1. Model processes visual features")
    print("2. Combines with question tokens")
    print("3. Uses continuous thoughts for reasoning")
    print("4. Generates answer tokens")
    
    # Mock generated response
    mock_response = "The cat is sleeping peacefully on a comfortable chair."
    print(f"\nGenerated Response: {mock_response}")
    
    print("\nAdvantages of Continuous Thought Inference:")
    print("• Faster generation (no explicit reasoning steps)")
    print("• More efficient token usage")
    print("• Maintains reasoning quality")
    print("• Scalable to complex reasoning tasks")


@tutorial_section("8. Evaluation and Metrics")
def section_8_evaluation():
    """
    Discuss evaluation methods and metrics
    """
    print("Evaluating Multimodal CoCoNuT involves multiple dimensions of performance.\n")
    
    print("Evaluation Metrics:")
    print("\n1. ACCURACY METRICS:")
    print("   • Exact Match: Percentage of exactly correct answers")
    print("   • BLEU Score: N-gram overlap with reference answers")
    print("   • ROUGE Score: Recall-oriented evaluation")
    print("   • CIDEr Score: Consensus-based evaluation")
    
    print("\n2. REASONING QUALITY:")
    print("   • Reasoning Path Accuracy: Quality of intermediate steps")
    print("   • Logical Consistency: Coherence of reasoning chain")
    print("   • Error Analysis: Types and frequency of mistakes")
    
    print("\n3. EFFICIENCY METRICS:")
    print("   • Inference Speed: Tokens per second")
    print("   • Memory Usage: Peak GPU memory consumption")
    print("   • Token Efficiency: Answer quality per token used")
    
    print("\n4. MULTIMODAL UNDERSTANDING:")
    print("   • Visual Grounding: Correct identification of visual elements")
    print("   • Cross-modal Reasoning: Integration of visual and textual information")
    print("   • Compositional Understanding: Complex scene interpretation")
    
    # Mock evaluation results
    print("\nSample Evaluation Results:")
    results = {
        "exact_match": 0.72,
        "bleu_4": 0.68,
        "rouge_l": 0.75,
        "inference_speed": 45.2,  # tokens/sec
        "memory_usage": 8.4,  # GB
        "reasoning_accuracy": 0.69
    }
    
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    
    print("\nComparison with Baselines:")
    print("• Standard Chain-of-Thought: More explicit but slower")
    print("• Direct Answer Generation: Faster but less accurate")
    print("• CoCoNuT: Balanced accuracy and efficiency")


@tutorial_section("9. Advanced Features")
def section_9_advanced_features():
    """
    Explore advanced features and capabilities
    """
    print("Advanced features extend the capabilities of Multimodal CoCoNuT.\n")
    
    print("1. DISTRIBUTED TRAINING:")
    print("   • FSDP (Fully Sharded Data Parallel)")
    print("   • DDP (Distributed Data Parallel)")
    print("   • Mixed Precision Training")
    print("   • Gradient Checkpointing")
    
    print("\n2. MEMORY OPTIMIZATION:")
    print("   • KV Cache Reuse: Efficient multi-pass processing")
    print("   • Gradient Accumulation: Large effective batch sizes")
    print("   • CPU Offloading: Handle larger models")
    print("   • Flash Attention: Efficient attention computation")
    
    print("\n3. CONFIGURATION FLEXIBILITY:")
    print("   • Template-based Configurations")
    print("   • Environment Variable Substitution")
    print("   • Stage-specific Parameter Updates")
    print("   • Validation and Error Checking")
    
    print("\n4. DEBUGGING AND MONITORING:")
    print("   • Comprehensive Logging")
    print("   • WandB Integration")
    print("   • Gradient Monitoring")
    print("   • Memory Profiling")
    
    print("\n5. EXTENSIBILITY:")
    print("   • Custom Dataset Support")
    print("   • Pluggable Components")
    print("   • Model Architecture Variants")
    print("   • Custom Evaluation Metrics")
    
    # Show configuration example for advanced features
    print("\nAdvanced Configuration Example:")
    advanced_config = {
        "use_fsdp": True,
        "gradient_checkpointing": True,
        "fp16": True,
        "gradient_accumulation_steps": 4,
        "use_flash_attention": True,
        "wandb_project": "multimodal-coconut",
        "save_every_n_epochs": 2,
        "eval_every_n_epochs": 1
    }
    
    for key, value in advanced_config.items():
        print(f"  {key}: {value}")


@tutorial_section("10. Best Practices and Tips")
def section_10_best_practices():
    """
    Share best practices and practical tips
    """
    print("Best practices for successful Multimodal CoCoNuT training and deployment.\n")
    
    print("TRAINING BEST PRACTICES:")
    print("\n1. Data Preparation:")
    print("   • Ensure high-quality image-question-answer triplets")
    print("   • Balance dataset across different question types")
    print("   • Validate data format and paths")
    print("   • Use appropriate train/validation splits")
    
    print("\n2. Model Configuration:")
    print("   • Start with pre-trained InternVL3 weights")
    print("   • Use appropriate batch sizes for your hardware")
    print("   • Set reasonable learning rates (1e-5 to 5e-6)")
    print("   • Enable gradient checkpointing for memory efficiency")
    
    print("\n3. Training Strategy:")
    print("   • Begin with CoT pre-training (Stage 0)")
    print("   • Gradually increase latent stages")
    print("   • Monitor validation metrics closely")
    print("   • Use early stopping to prevent overfitting")
    
    print("\n4. Hyperparameter Tuning:")
    print("   • c_thought: Start with 2, experiment with 3-4")
    print("   • max_latent_stage: Typically 4-6 stages")
    print("   • epochs_per_stage: 3-5 epochs usually sufficient")
    print("   • uniform_prob: 0.1 for balanced sampling")
    
    print("\nDEBUGGING TIPS:")
    print("\n1. Start Small:")
    print("   • Use debug configuration for initial testing")
    print("   • Test with small datasets first")
    print("   • Verify data loading pipeline")
    
    print("\n2. Monitor Training:")
    print("   • Watch for NaN losses (reduce learning rate)")
    print("   • Check gradient norms (clip if too large)")
    print("   • Monitor GPU memory usage")
    print("   • Validate special token handling")
    
    print("\n3. Common Issues:")
    print("   • CUDA OOM: Reduce batch size or image size")
    print("   • Slow training: Enable Flash Attention")
    print("   • Poor performance: Check data quality")
    print("   • Model divergence: Lower learning rate")
    
    print("\nDEPLOYMENT CONSIDERATIONS:")
    print("\n1. Model Optimization:")
    print("   • Use torch.compile() for faster inference")
    print("   • Enable KV caching for generation")
    print("   • Consider model quantization")
    
    print("\n2. Serving Setup:")
    print("   • Batch multiple requests together")
    print("   • Pre-compute visual features when possible")
    print("   • Implement proper error handling")
    print("   • Monitor inference latency and throughput")
    
    print("\n3. Quality Assurance:")
    print("   • Test on diverse image types")
    print("   • Validate reasoning quality")
    print("   • Monitor for bias and fairness")
    print("   • Implement safety filters")


def main():
    """Run the complete tutorial"""
    print("MULTIMODAL COCONUT COMPREHENSIVE TUTORIAL")
    print("This tutorial covers all aspects of the Multimodal CoCoNuT system.")
    print("Follow along to understand the concepts and implementation details.\n")
    
    try:
        # Run all tutorial sections
        section_1_introduction()
        config = section_2_configuration()
        model, tokenizer = section_3_model_architecture()
        image_processor = section_4_data_pipeline()
        section_5_training_process()
        section_6_continuous_thoughts()
        section_7_inference()
        section_8_evaluation()
        section_9_advanced_features()
        section_10_best_practices()
        
        print("\n" + "="*80)
        print("TUTORIAL COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nYou now have a comprehensive understanding of:")
        print("✓ Core CoCoNuT concepts and continuous thought mechanism")
        print("✓ Configuration system and template usage")
        print("✓ Model architecture and multimodal integration")
        print("✓ Data pipeline and preprocessing")
        print("✓ Staged training curriculum")
        print("✓ Inference and generation process")
        print("✓ Evaluation methods and metrics")
        print("✓ Advanced features and optimization")
        print("✓ Best practices and debugging tips")
        
        print("\nNext Steps:")
        print("1. Try the basic training example: python examples/basic_training.py")
        print("2. Experiment with configurations: python examples/config_examples.py")
        print("3. Run inference examples: python examples/inference_example.py")
        print("4. Read the API documentation: docs/API.md")
        print("5. Check troubleshooting guide: docs/TROUBLESHOOTING.md")
        
    except Exception as e:
        print(f"\nTutorial error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()