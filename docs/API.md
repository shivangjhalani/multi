# Multimodal CoCoNuT API Documentation

This document provides comprehensive API documentation for all major classes and functions in the Multimodal CoCoNuT project.

## Table of Contents

- [Configuration System](#configuration-system)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Training System](#training-system)
- [Utilities](#utilities)
- [Examples](#examples)

## Configuration System

### Config Class

The `Config` class provides a simple and elegant configuration management system following the original CoCoNuT patterns.

```python
from multimodal_coconut import Config, load_config
```

#### Class: `Config`

**Purpose**: Simple configuration class that allows accessing dictionary values as object attributes with enhanced validation.

**Constructor**:
```python
Config(config_dict: Dict[str, Any])
```

**Parameters**:
- `config_dict`: Dictionary containing configuration parameters

**Methods**:

##### `to_dict() -> Dict[str, Any]`
Convert configuration back to dictionary format.

**Returns**: Dictionary representation of the configuration

##### `update(**kwargs) -> None`
Update configuration with new values.

**Parameters**:
- `**kwargs`: Key-value pairs to update in the configuration

##### `get(key: str, default=None) -> Any`
Get configuration value with optional default.

**Parameters**:
- `key`: Configuration key to retrieve
- `default`: Default value if key doesn't exist

**Returns**: Configuration value or default

##### `has(key: str) -> bool`
Check if configuration has a specific key.

**Parameters**:
- `key`: Configuration key to check

**Returns**: True if key exists, False otherwise

##### `merge(other_config: Config) -> Config`
Merge with another configuration object.

**Parameters**:
- `other_config`: Another Config object to merge with

**Returns**: New Config object with merged settings

##### `save(path: Union[str, Path]) -> None`
Save configuration to YAML file.

**Parameters**:
- `path`: File path to save configuration

### Configuration Functions

#### `load_config(config_path: str, validate: bool = True) -> Config`

Load configuration from YAML file with environment variable substitution.

**Parameters**:
- `config_path`: Path to YAML configuration file
- `validate`: Whether to validate configuration after loading (default: True)

**Returns**: Config object with loaded settings

**Raises**: 
- `ConfigError`: If configuration loading or validation fails

**Example**:
```python
config = load_config('args/multimodal_coconut.yaml')
print(f"Model ID: {config.model_id}")
print(f"Batch size: {config.batch_size_training}")
```

#### `validate_config(config: Config) -> None`

Validate configuration parameters with comprehensive error checking.

**Parameters**:
- `config`: Configuration object to validate

**Raises**: 
- `ConfigError`: If configuration is invalid

**Validation Checks**:
- Required fields presence
- CoCoNuT parameter ranges
- Training parameter validity
- Multimodal parameter constraints
- Data path existence
- Distributed training settings

#### `create_config_from_template(template_name: str, **overrides) -> Config`

Create configuration from predefined templates.

**Parameters**:
- `template_name`: Template name ('default', 'cot', 'coconut', 'eval', 'debug')
- `**overrides`: Configuration values to override

**Returns**: Config object with template settings and overrides

**Available Templates**:
- `'default'`: Standard multimodal CoCoNuT configuration
- `'cot'`: Chain-of-Thought pre-training configuration
- `'coconut'`: CoCoNuT training configuration
- `'eval'`: Evaluation-only configuration
- `'debug'`: Debug configuration with small settings

**Example**:
```python
# Create debug configuration with custom batch size
config = create_config_from_template('debug', batch_size_training=4)
```

## Model Architecture

### MultimodalCoconut Class

The core model class that extends CoCoNuT methodology to multimodal reasoning.

```python
from multimodal_coconut.model import MultimodalCoconut
```

#### Class: `MultimodalCoconut(nn.Module)`

**Purpose**: Multimodal CoCoNuT model that extends the original CoCoNuT architecture to handle multimodal inputs using InternVL3 as the base model.

**Constructor**:
```python
MultimodalCoconut(
    base_model: nn.Module,
    latent_token_id: int,
    start_latent_id: int,
    end_latent_id: int,
    eos_token_id: int
)
```

**Parameters**:
- `base_model`: InternVL3 base model instance
- `latent_token_id`: Token ID for `<|latent|>` tokens
- `start_latent_id`: Token ID for `<|start-latent|>` tokens
- `end_latent_id`: Token ID for `<|end-latent|>` tokens
- `eos_token_id`: End of sequence token ID

**Key Attributes**:
- `base_model`: The underlying InternVL3 model
- `hidden_size`: Language model hidden dimension
- `latent_token_id`: ID for latent reasoning tokens

**Methods**:

##### `forward(...) -> CausalLMOutputWithPast`

Main forward pass with continuous thought reasoning.

**Parameters**:
- `input_ids` (torch.LongTensor): Text token IDs [batch_size, sequence_length]
- `pixel_values` (Optional[torch.FloatTensor]): Image pixel values [total_patches, channels, height, width]
- `attention_mask` (Optional[torch.Tensor]): Attention mask [batch_size, sequence_length]
- `position_ids` (Optional[torch.LongTensor]): Position IDs [batch_size, sequence_length]
- `image_flags` (Optional[torch.LongTensor]): Image presence flags [batch_size, 1]
- `past_key_values` (Optional[List[torch.FloatTensor]]): Cached key-value pairs
- `labels` (Optional[torch.LongTensor]): Target labels [batch_size, sequence_length]
- `use_cache` (Optional[bool]): Whether to use KV cache
- `output_attentions` (Optional[bool]): Whether to output attention weights
- `output_hidden_states` (Optional[bool]): Whether to output hidden states
- `return_dict` (Optional[bool]): Whether to return dictionary format

**Returns**: `CausalLMOutputWithPast` containing:
- `loss`: Training loss (if labels provided)
- `logits`: Model predictions [batch_size, sequence_length, vocab_size]
- `past_key_values`: Updated KV cache
- `hidden_states`: Hidden states (if requested)
- `attentions`: Attention weights (if requested)

**Algorithm**:
1. Detect latent token positions in input sequence
2. If no latent tokens: use standard multimodal forward pass
3. If latent tokens present: perform iterative CoCoNuT processing
4. Use KV cache for efficiency during multi-pass processing
5. Feed hidden states back as embeddings for continuous thoughts

##### `generate(...) -> torch.LongTensor`

Generate text with multimodal continuous thought reasoning.

**Parameters**:
- `pixel_values` (Optional[torch.FloatTensor]): Image pixel values
- `input_ids` (Optional[torch.LongTensor]): Input token IDs
- `attention_mask` (Optional[torch.Tensor]): Attention mask
- `image_flags` (Optional[torch.LongTensor]): Image presence flags
- `visual_features` (Optional[torch.FloatTensor]): Pre-computed visual features
- `generation_config` (Optional[dict]): Generation configuration
- `**generate_kwargs`: Additional generation arguments

**Returns**: Generated token IDs [batch_size, generated_length]

**Generation Parameters**:
- `max_new_tokens`: Maximum tokens to generate (default: 100)
- `do_sample`: Whether to use sampling (default: True)
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `top_k`: Top-k sampling parameter (default: 50)

**Example**:
```python
# Generate response for image-question pair
generated_ids = model.generate(
    pixel_values=images,
    input_ids=input_ids,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)
```

## Data Pipeline

### Dataset Classes

#### Class: `MultimodalDataset`

**Purpose**: Dataset class for loading and processing multimodal data (images + text) compatible with CoCoNuT training.

```python
from multimodal_coconut.data import MultimodalDataset
```

**Constructor**:
```python
MultimodalDataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    image_processor: ImageProcessor,
    max_size: Optional[int] = None,
    image_root: Optional[str] = None
)
```

**Parameters**:
- `data_path`: Path to JSON data file
- `tokenizer`: HuggingFace tokenizer instance
- `image_processor`: Image preprocessing instance
- `max_size`: Maximum dataset size (for debugging)
- `image_root`: Root directory for image files

**Methods**:

##### `__getitem__(idx: int) -> Dict[str, Any]`

Get a single data sample.

**Returns**: Dictionary containing:
- `pixel_values`: Processed image tensor
- `input_ids`: Tokenized text sequence
- `attention_mask`: Attention mask
- `labels`: Target labels for training
- `num_patches`: Number of image patches

##### `tokenize_multimodal_sample(sample: Dict) -> Dict[str, Any]`

Tokenize a multimodal sample for training.

**Parameters**:
- `sample`: Raw data sample with 'question', 'steps', 'answer' fields

**Returns**: Tokenized sample dictionary

#### Class: `MultimodalCollator`

**Purpose**: Custom data collator for efficient batching of multimodal data with latent token alignment.

```python
from multimodal_coconut.data import MultimodalCollator
```

**Constructor**:
```python
MultimodalCollator(
    tokenizer: AutoTokenizer,
    image_processor: ImageProcessor,
    latent_id: int
)
```

**Parameters**:
- `tokenizer`: HuggingFace tokenizer
- `image_processor`: Image processor instance
- `latent_id`: Token ID for latent tokens

**Methods**:

##### `__call__(features: List[Dict]) -> Dict[str, torch.Tensor]`

Collate a batch of features.

**Parameters**:
- `features`: List of sample dictionaries

**Returns**: Batched tensors dictionary containing:
- `pixel_values`: Batched image tensors
- `input_ids`: Padded token sequences
- `attention_mask`: Attention masks
- `labels`: Target labels
- `position_ids`: Position IDs
- `image_flags`: Image presence flags

### Image Processing

#### Class: `ImageProcessor`

**Purpose**: Wrapper around InternVL3's image preprocessing pipeline.

```python
from multimodal_coconut.data import ImageProcessor
```

**Constructor**:
```python
ImageProcessor(
    image_size: int = 448,
    max_num_patches: int = 12,
    use_thumbnail: bool = True,
    dynamic_preprocess: bool = True
)
```

**Parameters**:
- `image_size`: Target image size for processing
- `max_num_patches`: Maximum number of image patches
- `use_thumbnail`: Whether to use thumbnail preprocessing
- `dynamic_preprocess`: Whether to use dynamic preprocessing

**Methods**:

##### `process_image(image_path: str) -> torch.Tensor`

Process a single image file.

**Parameters**:
- `image_path`: Path to image file

**Returns**: Processed image tensor [num_patches, channels, height, width]

##### `process_images(image_paths: List[str]) -> torch.Tensor`

Process multiple images.

**Parameters**:
- `image_paths`: List of image file paths

**Returns**: Batched image tensors

## Training System

### Trainer Classes

#### Class: `MultimodalCoTTrainer`

**Purpose**: Main training orchestrator for multimodal CoCoNuT with staged curriculum learning.

```python
from multimodal_coconut.training import MultimodalCoTTrainer
```

**Constructor**:
```python
MultimodalCoTTrainer(
    model: MultimodalCoconut,
    tokenizer: AutoTokenizer,
    image_processor: ImageProcessor,
    config: Config
)
```

**Parameters**:
- `model`: MultimodalCoconut model instance
- `tokenizer`: HuggingFace tokenizer
- `image_processor`: Image processor instance
- `config`: Configuration object

**Methods**:

##### `train() -> None`

Execute the complete training loop with staged curriculum.

**Training Stages**:
- Stage 0: Standard multimodal chain-of-thought
- Stage k: First k reasoning steps replaced with latent tokens
- Progressive deepening through curriculum

##### `train_epoch(epoch: int, dataloader, optimizer) -> Dict[str, float]`

Train for a single epoch.

**Parameters**:
- `epoch`: Current epoch number
- `dataloader`: Training data loader
- `optimizer`: Optimizer instance

**Returns**: Dictionary with training metrics

##### `validate(dataloader) -> Dict[str, float]`

Run validation on provided data loader.

**Parameters**:
- `dataloader`: Validation data loader

**Returns**: Dictionary with validation metrics

##### `save_checkpoint(epoch: int, metrics: Dict[str, float]) -> None`

Save model checkpoint with training state.

**Parameters**:
- `epoch`: Current epoch number
- `metrics`: Current training metrics

#### Class: `StageManager`

**Purpose**: Manages curriculum progression through training stages.

```python
from multimodal_coconut.training import StageManager
```

**Constructor**:
```python
StageManager(
    max_latent_stage: int,
    epochs_per_stage: int,
    c_thought: int = 2
)
```

**Parameters**:
- `max_latent_stage`: Maximum number of latent stages
- `epochs_per_stage`: Number of epochs per curriculum stage
- `c_thought`: Number of continuous thoughts per reasoning step

**Methods**:

##### `get_current_stage(epoch: int) -> int`

Get current training stage based on epoch.

**Parameters**:
- `epoch`: Current epoch number

**Returns**: Current stage number (0 = CoT, >0 = CoCoNuT)

##### `prepare_stage_data(dataset, stage: int) -> Dataset`

Prepare dataset for specific training stage.

**Parameters**:
- `dataset`: Base dataset
- `stage`: Target training stage

**Returns**: Modified dataset with appropriate latent token replacements

##### `get_stage_config(stage: int) -> Dict[str, Any]`

Get configuration updates for specific stage.

**Parameters**:
- `stage`: Training stage number

**Returns**: Dictionary with stage-specific configuration updates

## Utilities

### Distributed Training

#### `init_distributed_training() -> None`

Initialize distributed training environment.

**Environment Variables**:
- `RANK`: Process rank
- `WORLD_SIZE`: Total number of processes
- `LOCAL_RANK`: Local process rank
- `MASTER_ADDR`: Master node address
- `MASTER_PORT`: Master node port

#### `is_main_process() -> bool`

Check if current process is the main process.

**Returns**: True if main process, False otherwise

### Logging

#### `setup_logging(config: Config) -> None`

Set up logging configuration with optional WandB integration.

**Parameters**:
- `config`: Configuration object

**Features**:
- Structured logging with timestamps
- WandB integration for experiment tracking
- Configurable log levels
- File and console output

#### `get_logger(name: str) -> logging.Logger`

Get a configured logger instance.

**Parameters**:
- `name`: Logger name (typically `__name__`)

**Returns**: Configured logger instance

### Miscellaneous

#### `set_seed(seed: int) -> None`

Set random seeds for reproducibility.

**Parameters**:
- `seed`: Random seed value

**Sets seeds for**:
- Python random module
- NumPy random
- PyTorch random
- CUDA random (if available)

## Examples

### Basic Usage

```python
from multimodal_coconut import Config, load_config, MultimodalCoconut
from multimodal_coconut.data import MultimodalDataset, MultimodalCollator
from multimodal_coconut.training import MultimodalCoTTrainer

# Load configuration
config = load_config('args/multimodal_coconut.yaml')

# Initialize model
model = MultimodalCoconut.from_pretrained(config.model_id, config)

# Create dataset
dataset = MultimodalDataset(
    data_path=config.train_data_path,
    tokenizer=tokenizer,
    image_processor=image_processor
)

# Create trainer
trainer = MultimodalCoTTrainer(model, tokenizer, image_processor, config)

# Start training
trainer.train()
```

### Custom Configuration

```python
from multimodal_coconut import create_config_from_template

# Create custom configuration
config = create_config_from_template(
    'coconut',
    c_thought=3,
    max_latent_stage=6,
    batch_size_training=4,
    learning_rate=5e-6
)

# Save custom configuration
config.save('args/my_experiment.yaml')
```

### Inference Example

```python
import torch
from PIL import Image

# Load model
model = MultimodalCoconut.from_pretrained('path/to/checkpoint')
model.eval()

# Prepare inputs
image = Image.open('path/to/image.jpg')
question = "What is happening in this image?"

# Process inputs
pixel_values = image_processor.process_image(image)
input_ids = tokenizer.encode(question, return_tensors='pt')

# Generate response
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=pixel_values.unsqueeze(0),
        input_ids=input_ids,
        max_new_tokens=100,
        temperature=0.7
    )

# Decode response
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"Response: {response}")
```

### Error Handling

```python
from multimodal_coconut.config import ConfigError

try:
    config = load_config('invalid_config.yaml')
except ConfigError as e:
    print(f"Configuration error: {e}")
    # Handle configuration error
    config = create_config_from_template('default')
```

## Error Handling and Troubleshooting

### Common Errors

#### ConfigError
Raised when configuration loading or validation fails.

**Common causes**:
- Missing required configuration fields
- Invalid parameter ranges
- File not found errors

**Solution**: Check configuration file syntax and required fields.

#### CUDA Out of Memory
Common during training with large batch sizes or images.

**Solutions**:
- Reduce `batch_size_training`
- Enable gradient checkpointing
- Use smaller `image_size`
- Reduce `max_num_patches`

#### Shape Mismatch Errors
Can occur during multimodal processing.

**Solutions**:
- Ensure consistent image preprocessing
- Check tokenizer compatibility
- Verify batch collation logic

### Performance Optimization

#### Memory Optimization
- Use gradient checkpointing: `config.gradient_checkpointing = True`
- Enable mixed precision: `config.fp16 = True`
- Reduce batch size for large images

#### Training Speed
- Use FSDP for large models: `config.use_fsdp = True`
- Optimize data loading: `config.num_workers = 4`
- Use efficient image preprocessing

#### Inference Optimization
- Use KV cache: `use_cache=True`
- Batch multiple samples together
- Pre-compute visual features when possible

This API documentation provides comprehensive coverage of all major classes and functions in the Multimodal CoCoNuT project. For additional examples and tutorials, see the examples directory and tutorial notebooks.