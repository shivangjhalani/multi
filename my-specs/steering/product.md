# Product Overview

## Multimodal CoCoNuT: Chain of Continuous Thought for Visual Question Answering

This project extends the CoCoNuT (Chain of Continuous Thought) methodology to multimodal reasoning, combining InternVL3's vision-language capabilities with continuous latent reasoning for visual question answering tasks.

### Core Innovation
- **Continuous Thought Mechanism**: Replaces discrete textual reasoning steps with continuous vector representations in a high-dimensional latent space
- **Multimodal Integration**: Built on InternVL3-1B-Pretrained for vision-language understanding
- **Staged Curriculum Learning**: Progressive training from explicit reasoning to latent thoughts

### Key Features
- Continuous latent reasoning using special `<|latent|>` tokens
- Staged training curriculum (Stage 0: CoT pre-training → Stage k: k reasoning steps replaced with latent tokens)
- Multimodal data pipeline for A-OKVQA dataset
- Distributed training support (FSDP/DDP)
- YAML-based configuration system following CoCoNuT patterns

### Development Status
Currently in active development with infrastructure and core components implemented. The project follows the original CoCoNuT's elegant simplicity and minimal implementation approach.

### Target Use Cases
- Visual question answering with transparent reasoning
- Multimodal reasoning research
- Continuous thought representation learning
- Vision-language model enhancement