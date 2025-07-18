# Multimodal CoCoNuT: Chain of Continuous Thought for Visual Question Answering

This project extends the CoCoNuT (Chain of Continuous Thought) methodology to multimodal reasoning, combining InternVL3's vision-language capabilities with continuous latent reasoning for visual question answering tasks.

## Overview

CoCoNuT represents a paradigm shift from discrete textual reasoning steps to continuous thought vectors, allowing models to reason in a high-dimensional latent space. This multimodal extension adapts this approach to handle image-text pairs, creating a system that can perform visual question answering through continuous latent reasoning.

## Key Features

- **Continuous Thought Mechanism**: Replaces discrete reasoning steps with continuous vector representations
- **Staged Curriculum Learning**: Progressive training from explicit reasoning to latent thoughts
- **Multimodal Integration**: Built on InternVL3-1B-Pretrained for vision-language understanding
- **Distributed Training**: Support for FSDP and DDP training across multiple GPUs
- **Flexible Configuration**: YAML-based configuration system following CoCoNuT patterns

## Installation

### Prerequisites

- **Python**: 3.8 or higher (3.9 recommended)
- **CUDA**: Compatible GPU with CUDA 11.8+ (for GPU acceleration)
- **Memory**: At least 16GB RAM, 8GB+ GPU memory recommended
- **Storage**: 10GB+ free space for models and data

### Quick Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd multimodal-coconut
```

2. **Create and activate environment**:
```bash
# Using conda (recommended)
conda create -n multimodal-coconut python=3.9 -y
conda activate multimodal-coconut

# Or using venv
python -m venv multimodal-coconut
source multimodal-coconut/bin/activate  # Linux/Mac
# multimodal-coconut\Scripts\activate  # Windows
```

3. **Install PyTorch** (choose based on your CUDA version):
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (not recommended for training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Install the package**:
```bash
pip install -e .
```

### Verify Installation

Run the infrastructure test to verify everything is working:
```bash
python test_infrastructure.py
```

Expected output:
```
✓ All imports successful
✓ Configuration system working
✓ Model architecture functional
✓ Data pipeline operational
✓ Training utilities ready
✓ Infrastructure test passed!
```

### Troubleshooting Installation

If you encounter issues, see our [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for common solutions.

**Common fixes**:
- **CUDA issues**: Check `nvidia-smi` and install matching PyTorch version
- **Memory errors**: Ensure sufficient RAM/GPU memory
- **Package conflicts**: Use a fresh virtual environment

## Quick Start

### Configuration

The project uses YAML configuration files following the original CoCoNuT patterns. Example configurations are provided in the `args/` directory:

- `args/multimodal_cot.yaml`: Chain-of-Thought pre-training configuration
- `args/multimodal_coconut.yaml`: CoCoNuT training configuration

### Training

1. **Stage 0 - CoT Pre-training**:
```bash
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/multimodal_cot.yaml
```

2. **CoCoNuT Training**:
```bash
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/multimodal_coconut.yaml
```

### Evaluation

```bash
# Set only_eval: true in your config file
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/multimodal_coconut_eval.yaml
```

## Project Structure

```
multimodal-coconut/
├── multimodal_coconut/          # Main package
│   ├── config/                  # Configuration management
│   ├── data/                    # Data loading and processing
│   ├── model/                   # Model implementations
│   ├── training/                # Training utilities
│   └── utils/                   # Utility functions
├── args/                        # Configuration files
├── reference/                   # Reference implementations
├── requirements.txt             # Dependencies
├── run.py                      # Main training script
└── test_infrastructure.py      # Infrastructure test script
```

## Core Concepts

### CoCoNuT Mechanism

The core innovation is replacing discrete textual reasoning steps with continuous vector representations:

1. **Latent Tokens**: Special `<|latent|>` tokens mark positions for continuous thoughts
2. **Hidden State Feedback**: Previous hidden states become input embeddings for latent tokens
3. **Iterative Processing**: Multiple forward passes handle the dependency chain
4. **KV Cache Optimization**: Efficient reuse of Key/Value matrices

### Staged Training

Training progresses through stages:
- **Stage 0**: Standard multimodal chain-of-thought
- **Stage k**: First k reasoning steps replaced with latent tokens
- **Progressive Deepening**: Gradual increase in continuous reasoning

## Configuration

Key configuration parameters:

```yaml
# CoCoNuT parameters
c_thought: 2                    # Continuous thoughts per reasoning step
max_latent_stage: 4            # Maximum latent stage
epochs_per_stage: 5            # Epochs per curriculum stage

# Model settings
model_id: "OpenGVLab/InternVL3-1B-Pretrained"
image_size: 448
max_num_patches: 12

# Training settings
batch_size_training: 8
learning_rate: 1e-5
num_epochs: 40
```

## Documentation

### API Reference
- [API Documentation](docs/API.md) - Comprehensive API documentation for all classes and functions
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Solutions for common issues and problems

### Key Components
- **Configuration System**: YAML-based configuration with validation and templates
- **Model Architecture**: MultimodalCoconut class extending InternVL3 with continuous thoughts
- **Data Pipeline**: Efficient multimodal data loading and preprocessing
- **Training System**: Staged curriculum learning with distributed training support
- **Utilities**: Logging, distributed training, and debugging tools

## Usage Examples

### Basic Training
```python
from multimodal_coconut import load_config, MultimodalCoconut
from multimodal_coconut.training import MultimodalCoTTrainer

# Load configuration
config = load_config('args/multimodal_coconut.yaml')

# Initialize model and trainer
model = MultimodalCoconut.from_pretrained(config.model_id, config)
trainer = MultimodalCoTTrainer(model, tokenizer, image_processor, config)

# Start training
trainer.train()
```

### Inference Example
```python
import torch
from PIL import Image
from multimodal_coconut import MultimodalCoconut

# Load trained model
model = MultimodalCoconut.from_pretrained('path/to/checkpoint')
model.eval()

# Process image and question
image = Image.open('example.jpg')
question = "What is happening in this image?"

# Generate response
with torch.no_grad():
    response = model.generate(
        pixel_values=process_image(image),
        input_ids=tokenize_text(question),
        max_new_tokens=100
    )

print(f"Answer: {decode_response(response)}")
```

### Custom Configuration
```python
from multimodal_coconut import create_config_from_template

# Create custom configuration
config = create_config_from_template(
    'coconut',
    c_thought=3,
    max_latent_stage=6,
    batch_size_training=4
)

# Save for later use
config.save('args/my_experiment.yaml')
```

## Development Status

This project is currently in active development. The infrastructure and core components have been implemented and tested:

✅ **Completed**:
- Configuration system with validation
- Model architecture (MultimodalCoconut)
- Data pipeline components
- Training infrastructure
- Distributed training support
- Comprehensive documentation
- Testing framework

🚧 **In Progress**:
- Full integration testing
- Performance optimization
- Advanced evaluation metrics

📋 **Planned**:
- Additional dataset support
- Model compression techniques
- Deployment utilities

## Contributing

This project follows the original CoCoNuT's elegant simplicity. When contributing:

1. Follow the existing code patterns and style
2. Keep implementations minimal and focused
3. Add comprehensive tests for new features
4. Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original CoCoNuT paper and implementation
- InternVL3 team for the base multimodal model
- A-OKVQA dataset creators

## Citation

If you use this work, please cite:

```bibtex
@article{coconut2024,
  title={CoCoNuT: Reasoning in a Continuous Latent Space},
  author={...},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```