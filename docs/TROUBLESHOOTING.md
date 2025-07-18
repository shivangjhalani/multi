# Troubleshooting Guide

This guide covers common issues and their solutions when working with Multimodal CoCoNuT.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Training Issues](#training-issues)
- [Memory Problems](#memory-problems)
- [Model Loading Errors](#model-loading-errors)
- [Data Pipeline Issues](#data-pipeline-issues)
- [Distributed Training Problems](#distributed-training-problems)
- [Performance Issues](#performance-issues)
- [Debugging Tips](#debugging-tips)

## Installation Issues

### Problem: PyTorch Installation Fails

**Symptoms**:
```bash
ERROR: Could not find a version that satisfies the requirement torch>=2.5.1
```

**Solutions**:
1. **Check CUDA compatibility**:
   ```bash
   nvidia-smi  # Check CUDA version
   ```

2. **Install PyTorch with correct CUDA version**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Use conda instead**:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### Problem: Flash Attention Installation Fails

**Symptoms**:
```bash
ERROR: Failed building wheel for flash-attn
```

**Solutions**:
1. **Install pre-compiled wheel**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Install from conda-forge**:
   ```bash
   conda install flash-attn -c conda-forge
   ```

3. **Skip Flash Attention** (performance impact):
   ```bash
   pip install -r requirements.txt --ignore-installed flash-attn
   ```

### Problem: InternVL3 Model Download Fails

**Symptoms**:
```bash
OSError: OpenGVLab/InternVL3-1B-Pretrained does not appear to be a valid git repo
```

**Solutions**:
1. **Check internet connection and HuggingFace access**:
   ```bash
   huggingface-cli login
   ```

2. **Use local model path**:
   ```yaml
   model_id: "/path/to/local/internvl3-1b"
   ```

3. **Download manually**:
   ```bash
   git lfs install
   git clone https://huggingface.co/OpenGVLab/InternVL3-1B-Pretrained
   ```

## Configuration Problems

### Problem: ConfigError - Missing Required Fields

**Symptoms**:
```bash
ConfigError: Missing required config field: model_id
```

**Solutions**:
1. **Check required fields in config**:
   ```yaml
   model_id: "OpenGVLab/InternVL3-1B-Pretrained"
   c_thought: 2
   ```

2. **Use template configuration**:
   ```python
   from multimodal_coconut import create_config_from_template
   config = create_config_from_template('default')
   ```

3. **Validate configuration**:
   ```python
   from multimodal_coconut import validate_config
   validate_config(config)
   ```

### Problem: Invalid Parameter Ranges

**Symptoms**:
```bash
ConfigError: c_thought must be >= 1
```

**Solutions**:
1. **Check parameter constraints**:
   - `c_thought >= 1`
   - `max_latent_stage >= 1`
   - `epochs_per_stage >= 1`
   - `uniform_prob` between 0.0 and 1.0
   - `batch_size_training > 0`
   - `learning_rate > 0`

2. **Use valid ranges**:
   ```yaml
   c_thought: 2
   max_latent_stage: 4
   epochs_per_stage: 5
   uniform_prob: 0.1
   batch_size_training: 8
   learning_rate: 1e-5
   ```

### Problem: Environment Variable Substitution Fails

**Symptoms**:
```bash
ConfigError: Environment variable not found: DATA_ROOT
```

**Solutions**:
1. **Set environment variables**:
   ```bash
   export DATA_ROOT="/path/to/data"
   export MODEL_PATH="/path/to/models"
   ```

2. **Use default values in config**:
   ```yaml
   train_data_path: "${DATA_ROOT:data/aokvqa}/train.json"
   image_root: "${DATA_ROOT:data/aokvqa}/images"
   ```

3. **Use absolute paths**:
   ```yaml
   train_data_path: "/absolute/path/to/train.json"
   image_root: "/absolute/path/to/images"
   ```

## Training Issues

### Problem: Training Crashes with CUDA Errors

**Symptoms**:
```bash
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions**:
1. **Enable CUDA error checking**:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python run.py args/multimodal_coconut.yaml
   ```

2. **Check for invalid token IDs**:
   ```python
   # Verify special token IDs are valid
   print(f"Vocab size: {tokenizer.vocab_size}")
   print(f"Latent token ID: {latent_token_id}")
   assert latent_token_id < tokenizer.vocab_size
   ```

3. **Reduce batch size**:
   ```yaml
   batch_size_training: 2  # Start small
   batch_size_eval: 4
   ```

### Problem: Loss is NaN or Infinite

**Symptoms**:
```bash
Training loss: nan
```

**Solutions**:
1. **Check learning rate**:
   ```yaml
   learning_rate: 1e-5  # Reduce if too high
   ```

2. **Enable gradient clipping**:
   ```yaml
   max_grad_norm: 1.0
   ```

3. **Use mixed precision carefully**:
   ```yaml
   fp16: false  # Disable if causing instability
   ```

4. **Check for invalid labels**:
   ```python
   # Ensure labels are in valid range
   assert torch.all(labels >= -100)
   assert torch.all(labels < tokenizer.vocab_size)
   ```

### Problem: Training Extremely Slow

**Symptoms**:
- Very slow iterations per second
- High GPU memory usage but low utilization

**Solutions**:
1. **Enable efficient attention**:
   ```yaml
   use_flash_attention: true
   ```

2. **Optimize data loading**:
   ```yaml
   num_workers: 4
   pin_memory: true
   prefetch_factor: 2
   ```

3. **Use gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 4
   batch_size_training: 2  # Effective batch size = 2 * 4 = 8
   ```

4. **Enable compilation** (PyTorch 2.0+):
   ```python
   model = torch.compile(model)
   ```

## Memory Problems

### Problem: CUDA Out of Memory

**Symptoms**:
```bash
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. **Reduce batch size**:
   ```yaml
   batch_size_training: 1
   batch_size_eval: 2
   ```

2. **Reduce image size**:
   ```yaml
   image_size: 224  # Instead of 448
   max_num_patches: 6  # Instead of 12
   ```

3. **Enable gradient checkpointing**:
   ```yaml
   gradient_checkpointing: true
   ```

4. **Use FSDP for large models**:
   ```yaml
   use_fsdp: true
   fsdp_config:
     sharding_strategy: "FULL_SHARD"
     cpu_offload: true
   ```

5. **Clear cache regularly**:
   ```python
   torch.cuda.empty_cache()
   ```

### Problem: CPU Memory Issues

**Symptoms**:
```bash
MemoryError: Unable to allocate array
```

**Solutions**:
1. **Reduce dataset size for debugging**:
   ```yaml
   max_train_samples: 1000
   max_val_samples: 100
   ```

2. **Use streaming datasets**:
   ```python
   dataset = load_dataset("json", data_files=data_path, streaming=True)
   ```

3. **Optimize data loading**:
   ```yaml
   num_workers: 2  # Reduce if too high
   ```

4. **Use memory mapping**:
   ```python
   dataset = dataset.map(preprocess_function, num_proc=1, load_from_cache_file=True)
   ```

## Model Loading Errors

### Problem: Model Architecture Mismatch

**Symptoms**:
```bash
RuntimeError: Error(s) in loading state_dict for MultimodalCoconut
```

**Solutions**:
1. **Check model configuration**:
   ```python
   # Ensure config matches saved model
   config = load_config('args/multimodal_coconut.yaml')
   model = MultimodalCoconut.from_pretrained(checkpoint_path, config)
   ```

2. **Load with strict=False**:
   ```python
   model.load_state_dict(checkpoint, strict=False)
   ```

3. **Check checkpoint compatibility**:
   ```python
   checkpoint = torch.load(checkpoint_path, map_location='cpu')
   print("Checkpoint keys:", list(checkpoint.keys()))
   ```

### Problem: Tokenizer Compatibility Issues

**Symptoms**:
```bash
KeyError: '<|latent|>'
```

**Solutions**:
1. **Add special tokens to tokenizer**:
   ```python
   special_tokens = ["<|latent|>", "<|start-latent|>", "<|end-latent|>"]
   tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
   model.resize_token_embeddings(len(tokenizer))
   ```

2. **Save and load tokenizer with model**:
   ```python
   tokenizer.save_pretrained(save_path)
   tokenizer = AutoTokenizer.from_pretrained(save_path)
   ```

3. **Check token IDs**:
   ```python
   latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
   print(f"Latent token ID: {latent_id}")
   ```

## Data Pipeline Issues

### Problem: Image Loading Fails

**Symptoms**:
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'image.jpg'
```

**Solutions**:
1. **Check image paths**:
   ```python
   import os
   image_path = os.path.join(config.image_root, sample['image'])
   assert os.path.exists(image_path), f"Image not found: {image_path}"
   ```

2. **Use absolute paths**:
   ```yaml
   image_root: "/absolute/path/to/images"
   ```

3. **Handle missing images gracefully**:
   ```python
   try:
       image = Image.open(image_path)
   except FileNotFoundError:
       # Use placeholder image or skip sample
       image = Image.new('RGB', (224, 224), color='white')
   ```

### Problem: Data Format Issues

**Symptoms**:
```bash
KeyError: 'question'
```

**Solutions**:
1. **Check data format**:
   ```json
   {
     "question": "What is in the image?",
     "steps": ["Step 1: Look at the image", "Step 2: Identify objects"],
     "answer": "A cat",
     "image": "image001.jpg"
   }
   ```

2. **Validate data structure**:
   ```python
   required_keys = ['question', 'steps', 'answer', 'image']
   for sample in data:
       for key in required_keys:
           assert key in sample, f"Missing key: {key}"
   ```

3. **Handle missing fields**:
   ```python
   question = sample.get('question', '')
   steps = sample.get('steps', [])
   answer = sample.get('answer', '')
   ```

### Problem: Batch Collation Errors

**Symptoms**:
```bash
RuntimeError: stack expects each tensor to be equal size
```

**Solutions**:
1. **Check tensor shapes**:
   ```python
   print("Input shapes:")
   for i, item in enumerate(batch):
       print(f"Item {i}: {item['input_ids'].shape}")
   ```

2. **Use proper padding**:
   ```python
   # Ensure consistent padding in collator
   input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
   ```

3. **Handle variable image sizes**:
   ```python
   # Ensure all images have same number of patches
   max_patches = max(item['num_patches'] for item in batch)
   ```

## Distributed Training Problems

### Problem: Distributed Training Hangs

**Symptoms**:
- Training starts but hangs at first iteration
- No error messages

**Solutions**:
1. **Check network configuration**:
   ```bash
   # Test connectivity between nodes
   ping <master_node_ip>
   ```

2. **Use correct environment variables**:
   ```bash
   export MASTER_ADDR="localhost"
   export MASTER_PORT="12355"
   export WORLD_SIZE=2
   export RANK=0  # Different for each process
   ```

3. **Use torchrun instead of manual setup**:
   ```bash
   torchrun --nnodes=1 --nproc_per_node=2 run.py args/config.yaml
   ```

4. **Enable debugging**:
   ```bash
   export NCCL_DEBUG=INFO
   export TORCH_DISTRIBUTED_DEBUG=DETAIL
   ```

### Problem: FSDP Initialization Fails

**Symptoms**:
```bash
RuntimeError: FSDP requires PyTorch >= 1.12
```

**Solutions**:
1. **Check PyTorch version**:
   ```python
   import torch
   print(torch.__version__)  # Should be >= 1.12
   ```

2. **Use compatible FSDP configuration**:
   ```yaml
   use_fsdp: true
   fsdp_config:
     sharding_strategy: "FULL_SHARD"
     mixed_precision: true
     cpu_offload: false
   ```

3. **Fall back to DDP**:
   ```yaml
   use_fsdp: false
   use_ddp: true
   ```

## Performance Issues

### Problem: Slow Training Speed

**Symptoms**:
- Low GPU utilization
- Slow iterations per second

**Solutions**:
1. **Profile training loop**:
   ```python
   with torch.profiler.profile() as prof:
       # Training step
       pass
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

2. **Optimize data loading**:
   ```yaml
   num_workers: 4
   pin_memory: true
   persistent_workers: true
   ```

3. **Use efficient attention**:
   ```yaml
   use_flash_attention: true
   ```

4. **Enable mixed precision**:
   ```yaml
   fp16: true
   ```

### Problem: Poor Model Performance

**Symptoms**:
- High training loss
- Poor validation accuracy

**Solutions**:
1. **Check learning rate schedule**:
   ```yaml
   learning_rate: 1e-5
   warmup_steps: 1000
   lr_scheduler_type: "cosine"
   ```

2. **Verify data preprocessing**:
   ```python
   # Check tokenization
   sample = dataset[0]
   print(tokenizer.decode(sample['input_ids']))
   ```

3. **Monitor gradient norms**:
   ```python
   total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   print(f"Gradient norm: {total_norm}")
   ```

4. **Use curriculum learning properly**:
   ```yaml
   max_latent_stage: 4
   epochs_per_stage: 5
   ```

## Debugging Tips

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable PyTorch debugging
import torch
torch.autograd.set_detect_anomaly(True)
```

### Use Smaller Datasets for Testing

```yaml
max_train_samples: 100
max_val_samples: 50
num_epochs: 2
```

### Check Model Outputs

```python
# Inspect model outputs
outputs = model(input_ids=input_ids, pixel_values=pixel_values)
print(f"Logits shape: {outputs.logits.shape}")
print(f"Loss: {outputs.loss}")
```

### Monitor GPU Memory

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Call before and after operations
print_gpu_memory()
```

### Use Gradient Checkpointing for Memory

```python
model.gradient_checkpointing_enable()
```

### Test Individual Components

```python
# Test tokenizer
tokens = tokenizer("Test question", return_tensors="pt")
print(f"Tokens: {tokens}")

# Test image processor
image = Image.open("test.jpg")
processed = image_processor.process_image(image)
print(f"Processed image shape: {processed.shape}")

# Test model forward pass
with torch.no_grad():
    outputs = model(**tokens)
    print(f"Model outputs: {outputs}")
```

### Common Environment Variables

```bash
# CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Memory debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Distributed training debugging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Reproducibility
export PYTHONHASHSEED=42
```

### Quick Fixes Checklist

1. **Reduce batch size** if getting OOM errors
2. **Check file paths** are correct and accessible
3. **Verify CUDA availability** with `torch.cuda.is_available()`
4. **Update PyTorch** to latest stable version
5. **Clear CUDA cache** with `torch.cuda.empty_cache()`
6. **Check configuration** with `validate_config(config)`
7. **Use debug configuration** for initial testing
8. **Monitor system resources** (RAM, GPU memory, disk space)
9. **Check network connectivity** for distributed training
10. **Verify data format** matches expected structure

If you encounter issues not covered in this guide, please check the project's GitHub issues or create a new issue with detailed error messages and system information.