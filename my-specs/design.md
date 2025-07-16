# Design Document

## Overview

This document outlines the design for extending CoCoNuT (Chain of Continuous Thought) methodology to multimodal reasoning using InternVL3-1B-Pretrained as the base model. The system will enable Large Language Models to reason in continuous latent space while processing both visual and textual information, creating a novel approach to multimodal visual question answering.

The design builds upon the original CoCoNuT framework's staged training curriculum while adapting it to handle image-text pairs from datasets like A-OKVQA. The key innovation lies in extending the continuous thought mechanism to work with multimodal hidden states that encode both visual and textual information. This implementation focuses purely on multimodal functionality without maintaining backward compatibility for text-only CoCoNuT.

## Architecture

### High-Level Architecture

The multimodal CoCoNuT system consists of four main components:

1. **Multimodal Data Pipeline**: Handles loading, preprocessing, and batching of image-text pairs
2. **Multimodal CoCoNuT Model**: Extends the original CoCoNuT architecture to work with InternVL3's vision-language capabilities
3. **Staged Training System**: Implements the curriculum learning approach for multimodal continuous reasoning
4. **Evaluation Framework**: Provides comprehensive benchmarking and analysis tools

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal CoCoNuT System                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data Pipeline │  │  CoCoNuT Model  │  │ Training System │ │
│  │                 │  │                 │  │                 │ │
│  │ • A-OKVQA       │  │ • InternVL3     │  │ • Staged Curr.  │ │
│  │ • Image Proc.   │  │ • Continuous    │  │ • Mixed Stages  │ │
│  │ • Text Tokenize │  │   Thoughts      │  │ • Validation    │ │
│  │ • Batch Collate │  │ • KV Cache      │  │ • Checkpointing │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Evaluation Framework                        │ │
│  │ • VQA Metrics • Reasoning Analysis • Efficiency Benchmarks │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Model Architecture Details

The multimodal CoCoNuT model extends the original CoCoNuT architecture by integrating InternVL3's vision-language capabilities:

```
Input: [Image] + [Question] + [<|latent|>] * k + [Remaining Steps] + [Answer]
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    InternVL3 Base Model                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Vision Encoder │  │  Text Embedder  │  │ Multimodal LLM  │ │
│  │                 │  │                 │  │                 │ │
│  │ • InternViT     │  │ • Token Embed   │  │ • Transformer   │ │
│  │ • Dynamic Tiles │  │ • Position Emb  │  │ • Cross-Attn    │ │
│  │ • Visual Tokens │  │ • Special Tokens│  │ • Hidden States │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                 CoCoNuT Extension Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  • Latent Token Detection                                       │
│  • Multimodal Hidden State Feedback                            │
│  • Iterative Forward Passes with KV Cache                      │
│  • Continuous Thought Chain Processing                         │
└─────────────────────────────────────────────────────────────────┘
                     ↓
Output: [Generated Response based on Multimodal Continuous Reasoning]
```

## Components and Interfaces

### 1. Multimodal Data Pipeline

**Purpose**: Handle loading, preprocessing, and batching of multimodal datasets.

**Key Classes**:
- `MultimodalDataset`: Extends the original CoCoNuT dataset class to handle image-text pairs
- `MultimodalCollator`: Custom collator for efficient batching of multimodal data
- `ImageProcessor`: Wrapper around InternVL3's image preprocessing pipeline

**Interfaces**:
```python
class MultimodalDataset:
    def __init__(self, data_path: str, tokenizer, image_processor, max_size: int)
    def __getitem__(self, idx: int) -> Dict[str, Any]
    def tokenize_multimodal_sample(self, sample: Dict) -> Dict[str, Any]

class MultimodalCollator:
    def __init__(self, tokenizer, image_processor, latent_id: int)
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]
    def align_latent_tokens(self, batch: List[Dict]) -> Dict[str, torch.Tensor]
```

### 2. Multimodal CoCoNuT Model

**Purpose**: Extend CoCoNuT's continuous reasoning to multimodal inputs.

**Key Classes**:
- `MultimodalCoconut`: Main model class extending the original Coconut class
- `MultimodalForwardPass`: Handles the iterative forward pass logic for multimodal inputs

**Interfaces**:
```python
class MultimodalCoconut(nn.Module):
    def __init__(self, base_model, latent_token_id: int, start_latent_id: int, 
                 end_latent_id: int, eos_token_id: int)
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs)
    def multimodal_forward_pass(self, pixel_values: torch.Tensor, 
                               inputs_embeds: torch.Tensor, **kwargs)
    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, **kwargs)
```

### 3. Training System

**Purpose**: Implement staged curriculum learning for multimodal continuous reasoning.

**Key Classes**:
- `MultimodalTrainer`: Orchestrates the staged training process
- `StageManager`: Manages progression through training stages
- `ValidationRunner`: Handles periodic evaluation during training

**Interfaces**:
```python
class MultimodalTrainer:
    def __init__(self, model, tokenizer, image_processor, config)
    def train_epoch(self, epoch: int, dataloader, optimizer)
    def validate(self, dataloader) -> Dict[str, float]
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float])

class StageManager:
    def get_current_stage(self, epoch: int) -> int
    def prepare_stage_data(self, dataset, stage: int) -> Dataset
    def get_stage_config(self, stage: int) -> Dict[str, Any]
```

## Data Models

### Input Data Structure

```python
@dataclass
class MultimodalSample:
    image_path: str
    question: str
    reasoning_steps: List[str]
    answer: str
    idx: int
    
    # Processed fields
    pixel_values: Optional[torch.Tensor] = None
    question_tokenized: Optional[List[int]] = None
    steps_tokenized: Optional[List[List[int]]] = None
    answer_tokenized: Optional[List[int]] = None
```

### Training Batch Structure

```python
@dataclass
class MultimodalBatch:
    pixel_values: torch.Tensor  # [batch_size, num_patches, channels, height, width]
    input_ids: torch.Tensor     # [batch_size, sequence_length]
    attention_mask: torch.Tensor # [batch_size, sequence_length]
    labels: torch.Tensor        # [batch_size, sequence_length]
    position_ids: torch.Tensor  # [batch_size, sequence_length]
    num_patches_list: List[int] # Number of image patches per sample
```

### Configuration System

Following the original CoCoNuT's elegant YAML-based configuration approach, the multimodal extension uses a simple, hierarchical configuration system:

**YAML Configuration Structure**:
```yaml
# multimodal_coconut.yaml
name: "multimodal-coconut-aokvqa"
seed: 42

# Model configuration
model_id: "OpenGVLab/InternVL3-1B-Pretrained"
load_model_path: "None"  # Path to pre-trained checkpoint
coconut: true
cot: false

# CoCoNuT parameters
c_thought: 2
max_latent_stage: 4
epochs_per_stage: 5
uniform_prob: 0.1

# Training parameters
num_epochs: 40
batch_size_training: 8
batch_size_eval: 16
learning_rate: 1e-5
weight_decay: 0.01
warmup_steps: 1000

# Multimodal parameters
image_size: 448
max_num_patches: 12
use_thumbnail: true
dynamic_preprocess: true

# Data paths
train_data_path: "data/aokvqa/train.json"
val_data_path: "data/aokvqa/val.json"
test_data_path: "data/aokvqa/test.json"
image_root: "data/aokvqa/images"

# Distributed training
use_fsdp: true
use_ddp: false

# Evaluation
only_eval: false
eval_every_n_epochs: 5

# Checkpointing
save_path: "checkpoints"
resume: 0
save_every_n_epochs: 5
```

**Python Configuration Handler**:
```python
class Config:
    """Simple configuration class that mirrors the original CoCoNuT approach"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Config({self.__dict__})"
    
    def to_dict(self):
        return self.__dict__
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Usage example
with open('args/multimodal_coconut.yaml') as f:
    config_dict = yaml.safe_load(f)
config = Config(config_dict)
```

**Enhanced Configuration Features**:

1. **Environment Variable Support**:
```python
def load_config_with_env(config_path):
    """Load config with environment variable substitution"""
    with open(config_path) as f:
        config_str = f.read()
    
    # Replace environment variables
    import os
    for key, value in os.environ.items():
        config_str = config_str.replace(f'${{{key}}}', value)
    
    return yaml.safe_load(config_str)
```

2. **Configuration Validation**:
```python
def validate_config(config):
    """Validate configuration parameters"""
    required_fields = ['model_id', 'train_data_path', 'c_thought']
    for field in required_fields:
        if not hasattr(config, field):
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate ranges
    if config.c_thought < 1:
        raise ValueError("c_thought must be >= 1")
    if config.max_latent_stage < 1:
        raise ValueError("max_latent_stage must be >= 1")
```

3. **Configuration Inheritance**:
```yaml
# base_config.yaml
base: &base
  seed: 42
  learning_rate: 1e-5
  batch_size_training: 8

# experiment_config.yaml
<<: *base
name: "experiment-1"
c_thought: 3
max_latent_stage: 6
```

4. **Runtime Configuration Updates**:
```python
def update_config_for_stage(config, stage):
    """Update configuration based on training stage"""
    if stage == 0:  # CoT pre-training
        config.coconut = False
        config.cot = True
    else:  # CoCoNuT training
        config.coconut = True
        config.cot = False
    return config
```

## Error Handling

### Data Loading Errors

1. **Image Loading Failures**:
   - Implement fallback mechanisms for corrupted images
   - Log failed samples and continue training
   - Provide image validation during dataset preparation

2. **Tokenization Issues**:
   - Handle sequences that exceed maximum length
   - Implement truncation strategies for long reasoning chains
   - Validate special token integration

### Model Errors

1. **Memory Management**:
   - Implement gradient checkpointing for large models
   - Handle CUDA out-of-memory errors gracefully
   - Provide automatic batch size reduction

2. **Forward Pass Failures**:
   - Validate tensor shapes at each stage
   - Handle edge cases in latent token processing
   - Implement recovery mechanisms for KV cache issues

### Training Errors

1. **Distributed Training Issues**:
   - Handle node failures in multi-GPU setups
   - Implement checkpoint recovery mechanisms
   - Synchronize training state across processes

2. **Convergence Problems**:
   - Monitor training metrics for anomalies
   - Implement early stopping mechanisms
   - Provide learning rate scheduling options

## Testing Strategy

### Unit Testing

1. **Data Pipeline Tests**:
   - Test image preprocessing consistency
   - Validate tokenization correctness
   - Check batch collation logic

2. **Model Component Tests**:
   - Test multimodal forward pass logic
   - Validate KV cache functionality
   - Check latent token processing

3. **Training System Tests**:
   - Test stage progression logic
   - Validate checkpoint saving/loading
   - Check distributed training setup

### Integration Testing

1. **End-to-End Training**:
   - Test complete training pipeline on small dataset
   - Validate model convergence on toy examples
   - Check evaluation metrics computation

2. **Model Compatibility**:
   - Test with different InternVL3 model sizes
   - Validate backward compatibility with original CoCoNuT
   - Check integration with different datasets

### Performance Testing

1. **Memory Usage**:
   - Profile memory consumption during training
   - Test with different batch sizes and image resolutions
   - Validate KV cache efficiency

2. **Training Speed**:
   - Benchmark training throughput
   - Compare with baseline InternVL3 training
   - Test scaling across multiple GPUs

### Validation Testing

1. **Reasoning Quality**:
   - Test on held-out validation sets
   - Compare reasoning quality across stages
   - Validate continuous thought effectiveness

2. **Multimodal Understanding**:
   - Test on diverse image types and questions
   - Validate cross-modal reasoning capabilities
   - Check robustness to image quality variations

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Set up project structure and dependencies
- Implement basic multimodal data pipeline
- Create InternVL3 integration layer
- Develop initial testing framework

### Phase 2: Core Model (Weeks 3-4)
- Implement MultimodalCoconut class
- Develop multimodal forward pass logic
- Integrate KV cache for multimodal inputs
- Create basic training loop

### Phase 3: Training System (Weeks 5-6)
- Implement staged curriculum learning
- Develop validation and evaluation systems
- Add checkpoint management
- Integrate distributed training support

### Phase 4: Optimization (Weeks 7-8)
- Optimize memory usage and training speed
- Implement advanced batching strategies
- Add comprehensive error handling
- Develop debugging and monitoring tools

### Phase 5: Evaluation (Weeks 9-10)
- Conduct comprehensive benchmarking
- Compare with baseline models
- Analyze reasoning quality and efficiency
- Prepare documentation and examples

This design provides a comprehensive foundation for implementing multimodal CoCoNuT while maintaining compatibility with the original framework and leveraging InternVL3's powerful vision-language capabilities.