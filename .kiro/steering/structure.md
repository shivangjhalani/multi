# Project Structure & Organization

## Directory Layout

```
multimodal-coconut/
├── .kiro/                          # Kiro IDE configuration
│   ├── specs/                      # Project specifications
│   └── steering/                   # AI assistant steering rules
├── multimodal_coconut/             # Main package (core implementation)
│   ├── __init__.py                 # Package initialization
│   ├── config/                     # Configuration management
│   │   ├── __init__.py
│   │   └── config.py              # Config class and utilities
│   ├── data/                       # Data pipeline components
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset classes and collators
│   │   ├── dataset_utils.py       # Dataset utility functions
│   │   ├── image_processor.py     # Image preprocessing
│   │   └── prepare_aokvqa.py      # A-OKVQA dataset preparation
│   ├── model/                      # Model implementations
│   │   ├── __init__.py
│   │   └── multimodal_coconut.py  # Core CoCoNuT model
│   ├── training/                   # Training infrastructure
│   │   ├── __init__.py
│   │   ├── multimodal_cot_trainer.py  # Training orchestration
│   │   └── stage_manager.py       # Curriculum stage management
│   └── utils/                      # Utility modules
│       ├── __init__.py
│       ├── distributed.py         # Distributed training utilities
│       ├── logging.py             # Logging configuration
│       └── misc.py                # Miscellaneous helpers
├── args/                           # Configuration files
│   ├── multimodal_coconut.yaml    # CoCoNuT training config
│   └── multimodal_cot.yaml        # CoT pre-training config
├── reference/                      # Reference implementations
│   ├── InternVL/                   # InternVL reference code
│   └── coconut/                    # Original CoCoNuT reference
├── test_*.py                       # Test files (root level)
├── run.py                          # Main training script
├── setup.py                       # Package setup
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

## Core Components

### Main Package (`multimodal_coconut/`)
- **Modular Design**: Each subdirectory handles a specific aspect (config, data, model, training, utils)
- **Clean Interfaces**: Well-defined APIs between components
- **Minimal Dependencies**: Each module imports only what it needs

### Configuration System (`config/`)
- **Single Source of Truth**: All configuration through YAML files
- **Environment Integration**: Support for environment variable substitution
- **Validation**: Type checking and parameter validation
- **Stage Management**: Configuration updates based on training stage

### Data Pipeline (`data/`)
- **Dataset Classes**: `MultimodalDataset` for loading A-OKVQA data
- **Collators**: `MultimodalCollator` for efficient batching
- **Image Processing**: `ImageProcessor` wrapping InternVL3 preprocessing
- **Data Preparation**: Scripts for dataset setup and formatting

### Model Architecture (`model/`)
- **Core Model**: `MultimodalCoconut` class extending InternVL3
- **Continuous Thoughts**: Implementation of latent token processing
- **KV Cache**: Efficient multi-pass forward computation
- **Generation**: Inference methods for evaluation

### Training System (`training/`)
- **Trainer**: `MultimodalCoTTrainer` orchestrating training loops
- **Stage Manager**: `StageManager` handling curriculum progression
- **Distributed Support**: FSDP/DDP integration
- **Checkpointing**: Model state management

### Utilities (`utils/`)
- **Distributed**: Multi-GPU training setup
- **Logging**: Structured logging with WandB integration
- **Miscellaneous**: Helper functions and utilities

## File Naming Conventions

### Python Files
- **Snake Case**: `multimodal_coconut.py`, `stage_manager.py`
- **Descriptive Names**: Clear indication of purpose
- **Test Prefix**: `test_*.py` for all test files

### Configuration Files
- **Descriptive YAML**: `multimodal_coconut.yaml`, `multimodal_cot.yaml`
- **Stage Indication**: Clear naming for different training stages
- **Environment Specific**: Separate configs for different setups

### Documentation
- **Markdown Format**: `.md` extension for all documentation
- **Hierarchical**: README.md at root, specific docs in subdirectories
- **Comprehensive**: Detailed explanations with code examples

## Import Patterns

### Package Imports
```python
# Main package imports
from multimodal_coconut import Config, load_config
from multimodal_coconut.model import MultimodalCoconut
from multimodal_coconut.data import MultimodalDataset
```

### Relative Imports
```python
# Within package
from .config import Config
from ..utils import setup_logging
```

### External Dependencies
```python
# Standard library first
import os
import sys
from pathlib import Path

# Third-party libraries
import torch
import numpy as np
from transformers import AutoTokenizer

# Local imports last
from multimodal_coconut import Config
```

## Testing Structure

### Test Organization
- **Root Level**: All test files at project root for easy discovery
- **Descriptive Names**: `test_comprehensive_multimodal_coconut.py`
- **Hierarchical Testing**: Infrastructure → Integration → End-to-end

### Test Categories
- **Infrastructure Tests**: Basic functionality and imports
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Comprehensive Tests**: Full pipeline validation

## Development Workflow

### Code Organization Principles
1. **Separation of Concerns**: Each module has a single responsibility
2. **Minimal Coupling**: Loose dependencies between components
3. **Clear Interfaces**: Well-defined APIs and data contracts
4. **Consistent Patterns**: Following established conventions throughout

### File Creation Guidelines
- **Single Purpose**: One main class or function per file
- **Clear Dependencies**: Explicit imports and requirements
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Corresponding test file for each module