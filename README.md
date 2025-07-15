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

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 2.5.1 or higher

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multimodal-coconut
```

2. Create a virtual environment:
```bash
conda create -n multimodal-coconut python=3.9 -y
conda activate multimodal-coconut
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Verify Installation

Run the infrastructure test to verify everything is working:
```bash
python test_infrastructure.py
```

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

## Development Status

This project is currently in development. The infrastructure and core components have been implemented, with the following tasks remaining:

- [ ] Multimodal data pipeline implementation
- [ ] CoCoNuT model architecture completion
- [ ] Training loop implementation
- [ ] Evaluation system
- [ ] Comprehensive testing

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