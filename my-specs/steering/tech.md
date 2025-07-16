# Technology Stack & Build System

## Core Technologies

### Deep Learning Framework
- **PyTorch 2.5.1+**: Primary deep learning framework
- **Transformers 4.46.2+**: HuggingFace transformers for model loading
- **InternVL3-1B-Pretrained**: Base multimodal model from OpenGVLab

### Multimodal Processing
- **Pillow 10.0.0+**: Image processing and manipulation
- **OpenCV 4.8.0+**: Computer vision operations
- **timm 0.9.0+**: Vision model utilities

### Training Infrastructure
- **Flash Attention 2.3.6+**: Efficient attention computation
- **Accelerate 0.20.0+**: Distributed training utilities
- **DeepSpeed 0.10.0+**: Memory optimization and distributed training
- **FSDP/DDP**: Distributed training strategies

### Configuration & Data
- **PyYAML 6.0+**: YAML configuration parsing
- **OmegaConf 2.3.0+**: Advanced configuration management
- **Hydra Core 1.3.0+**: Configuration composition
- **Datasets 3.1.0+**: HuggingFace datasets library

### Development Tools
- **pytest 7.4.0+**: Testing framework
- **black 23.0.0+**: Code formatting
- **isort 5.12.0+**: Import sorting
- **flake8 6.0.0+**: Linting

## Build System

### Installation
```bash
# Create environment
conda create -n multimodal-coconut python=3.9 -y
conda activate multimodal-coconut

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Common Commands

#### Training
```bash
# Stage 0 - CoT Pre-training
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/multimodal_cot.yaml

# CoCoNuT Training
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/multimodal_coconut.yaml
```

#### Testing
```bash
# Infrastructure test
python test_infrastructure.py

# Comprehensive integration test
python test_comprehensive_multimodal_coconut.py

# Run all tests
pytest
```

#### Development
```bash
# Format code
black .
isort .

# Lint code
flake8 .

# Verify installation
python test_infrastructure.py
```

## Configuration System

### YAML-based Configuration
- Configuration files in `args/` directory
- Environment variable substitution supported (`${VAR_NAME}`)
- Stage-specific configuration updates
- Validation with type checking

### Key Configuration Patterns
- `coconut: true/false` - Enable CoCoNuT training
- `cot: true/false` - Enable CoT pre-training
- `c_thought: N` - Number of continuous thoughts per reasoning step
- `max_latent_stage: N` - Maximum latent stage for curriculum
- `epochs_per_stage: N` - Epochs per curriculum stage

## Distributed Training

### Supported Strategies
- **FSDP** (Fully Sharded Data Parallel): Recommended for large models
- **DDP** (Distributed Data Parallel): Alternative distributed strategy

### Configuration
```yaml
use_fsdp: true
use_ddp: false
```

## Development Patterns

### Code Style
- Follow original CoCoNuT's elegant simplicity
- Minimal implementations focused on core functionality
- Comprehensive docstrings and type hints
- Consistent naming conventions

### Testing Strategy
- Infrastructure tests for basic functionality
- Integration tests for component interaction
- Comprehensive end-to-end tests
- Mock data for testing without full datasets