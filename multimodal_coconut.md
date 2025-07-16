# Multimodal CoCoNuT: Reasoning in a Continuous Multimodal Latent Space

This document provides a detailed explanation of the Multimodal CoCoNuT (Chain of Continuous Thought) methodology. This project extends the original CoCoNuT framework to train Large Language Models (LLMs) for reasoning in a continuous latent space that incorporates both visual and textual information. We will delve into the core concepts, the adapted training algorithms, and the overall architecture, using InternVL3-1B-Pretrained as the base model.

## 1. Core Concepts

The fundamental innovation of CoCoNuT is the shift from discrete textual reasoning steps to continuous thought vectors. Multimodal CoCoNuT applies this concept to the vision-language domain.

### From Chain-of-Thought to Multimodal CoCoNuT

*   **Standard Multimodal Models:** These models take an image and a question as input and directly generate an answer, often lacking transparent reasoning steps.
*   **Multimodal Chain-of-Thought (M-CoT):** An extension of CoT, where the model generates explicit reasoning steps as discrete text tokens, conditioned on both the image and the question. The process is autoregressive, creating a textual chain of thoughts.
*   **Multimodal Chain of Continuous Thought (CoCoNuT):** This is the core of our project. Instead of generating textual reasoning steps, the model's internal multimodal hidden state is used as a "continuous thought." This continuous vector, which encodes both visual and textual context, is directly fed back into the model as the input embedding for the next reasoning step. This enables a more fluid and expressive reasoning process within a continuous latent space.

The diagram below illustrates the high-level architecture of the Multimodal CoCoNuT system.

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

## 2. Training Methodology

The training process is a staged curriculum designed to gradually transition the model from discrete multimodal reasoning to continuous thought.

### 2.1. M-CoT Training (Stage 0)

The foundation of a Multimodal CoCoNuT model is a model first trained with standard Multimodal Chain-of-Thought.

**Algorithm:**

1.  **Data Preparation:** The training data consists of `(image, question, steps, answer)` tuples from a dataset like A-OKVQA.
2.  **Input Formatting:** The input to the model is the image, followed by a concatenation of the question and the ground-truth textual reasoning steps.
3.  **Training Objective:** The model is trained to predict the next token in the sequence, including both the reasoning steps and the final answer, based on the visual and preceding textual context. The loss is a standard cross-entropy loss.
4.  **Configuration:** This initial stage is configured by setting `cot: True` and `coconut: False` in the training configuration, or by running a dedicated M-CoT pre-training script.

This stage teaches the model the fundamental reasoning patterns for the multimodal task.

### 2.2. The Multimodal CoCoNuT Training Curriculum

The curriculum gradually weans the model off explicit textual reasoning steps, encouraging it to represent those steps in its internal, continuous, multimodal space. This is achieved by progressively replacing ground-truth textual steps with special `<|latent|>` tokens.

**The Stages of Training:**

The training is divided into stages, with each stage lasting for `epochs_per_stage` epochs. The current stage (`scheduled_stage`) is calculated as `epoch // epochs_per_stage`.

*   **Stage 0 (M-CoT Pre-training):** As described above, this teaches the model basic multimodal reasoning.

*   **Stage 1:**
    *   **Goal:** Introduce the concept of a single continuous multimodal thought.
    *   **Data:** The *first* textual reasoning step is replaced by `c_thought` number of `<|latent|>` tokens. The model must then generate the rest of the reasoning steps and the final answer.
    *   **Example:**
        *   **Original Input:** `[Image] [Question] [Step 1] [Step 2] [Answer]`
        *   **Stage 1 Input:** `[Image] [Question] <|latent|> <|latent|> [Step 2] [Answer]` (for `c_thought=2`)
    *   **Learning Objective:** The model must learn to encode the information from "Step 1" into the continuous thought vectors, which are multimodal hidden states. This latent representation must be sufficient to correctly generate "Step 2".

*   **Stage 2 and Beyond:**
    *   **Goal:** Increase reliance on the internal reasoning process.
    *   **Data:** In each subsequent stage, one more textual reasoning step is replaced by `<|latent|>` tokens.
    *   **Progressive Deepening:** This continues up to `max_latent_stage`. By the final stage, the model performs the initial reasoning steps entirely in the continuous latent space.

This staged curriculum provides a stable learning signal, allowing the model to build upon its knowledge from previous stages and develop a robust internal representation of the multimodal reasoning process.

## 3. Architecture Deep-Dive

The system is built upon the powerful InternVL3 model, extending it with the CoCoNuT mechanism.

### Model Architecture Details

The core of the model is the integration of the CoCoNuT extension layer with the InternVL3 base model.

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

### Key Components and Interfaces

1.  **Multimodal Data Pipeline:**
    *   `MultimodalDataset`: Handles loading and tokenizing image-text pairs.
    *   `MultimodalCollator`: Efficiently batches multimodal data, aligning latent tokens for optimal performance.
    *   `ImageProcessor`: Wraps InternVL3's image preprocessing pipeline.

2.  **Multimodal CoCoNuT Model:**
    *   `MultimodalCoconut`: The main model class. It wraps the InternVL3 base model and implements the continuous thought feedback loop. Its `forward` method is the core of the multimodal reasoning process.

3.  **Training System:**
    *   `MultimodalTrainer`: Orchestrates the entire training process, including the staged curriculum.
    *   `StageManager`: Manages the progression through training stages, preparing the data accordingly.

## 4. Codebase Deep-Dive

The project follows the structure of the original CoCoNuT implementation, adapted for the multimodal context.

*   **`run.py`:** The main script for training and evaluation. It manages the staged training loop, initializes the distributed environment, and handles checkpointing.
*   **`multimodal_coconut/model/multimodal_coconut.py`:** Contains the `MultimodalCoconut` class. The `forward` method here implements the iterative, multi-pass logic for processing sequences with latent tokens, using multimodal hidden states for the continuous thought feedback.
*   **`multimodal_coconut/data/dataset.py`:** Contains the `MultimodalDataset` and `MultimodalCollator`. It prepares the data for each stage of the curriculum, replacing textual steps with `<|latent|>` tokens as required.
*   **`args/multimodal_coconut.yaml`:** The configuration file for training the Multimodal CoCoNuT model, specifying hyperparameters like `c_thought`, `epochs_per_stage`, `max_latent_stage`, and the path to the pre-trained M-CoT model.

## 5. Algorithms and Function Descriptions

The core algorithms from CoCoNuT are adapted to handle the additional image modality.

### 5.1. Multimodal CoT Training (Stage 0)

This is a standard multimodal auto-regressive language modeling task.

```
1. For each (image, question, steps, answer) in the training data:
2.   Preprocess the image to get `pixel_values`.
3.   Concatenate and tokenize the text:
     `input_ids = tokenize(question + "
" + "
".join(steps) + "
### " + answer)`
4.   For each batch:
5.     Perform a forward pass: `outputs = model(pixel_values, input_ids, labels=labels)`
6.     Calculate loss, backpropagate, and update weights.
```

### 5.2. Multimodal CoCoNuT Training

This implements the staged curriculum with the continuous thought mechanism.

```
1. Load the pre-trained M-CoT model.
2. For each training epoch:
3.   Determine the current stage: `scheduled_stage = epoch // epochs_per_stage`
4.   Prepare the input sequence by replacing the first `k` steps with `k * c_thought` latent tokens.
5.   For each batch:
6.     **Forward Pass (`MultimodalCoconut.forward`):**
7.       The input sequence is processed token by token, conditioned on the `pixel_values`.
8.       When a `latent_id` is encountered at position `i`:
9.         a. The forward pass is paused.
10.        b. The multimodal hidden state vector `h` from the output of the previous position (`i-1`) is retrieved.
11.        c. This hidden state `h` is used as the input embedding for the current position `i`.
12.        d. The forward pass resumes, using a `kv_cache` that contains context from both visual and previous text tokens.
13.    Calculate loss only on the remaining textual steps and the answer.
14.    Perform a backward pass and update weights.
```

## 6. Critical Implementation Details

*   **The Multi-Pass `forward` Method:** The core of the implementation. Since the input for a latent token depends on the output of the previous token, the `forward` method performs iterative passes, using the `kv_cache` to maintain efficiency. This is the direct implementation of the "chain" of continuous thoughts in a multimodal context.
*   **Multimodal `kv_cache`:** The Key-Value cache must efficiently store and retrieve context from both the visual tokens (from the image encoder) and the textual tokens. This is critical for making the multi-pass approach feasible.
*   **`MultimodalCollator`:** This custom collator is crucial for performance. It intelligently pads sequences to align the latent tokens across the batch, maximizing the reuse of the `kv_cache` during the initial part of the forward pass.

## 7. Practical Replication Guide

1.  **Environment and Data Setup:**
    *   Set up the environment using `requirements.txt`.
    *   Prepare the A-OKVQA dataset using the provided scripts, which will format it into the required JSON structure.

2.  **M-CoT Pre-training (Stage 0):**
    *   Configure and run the M-CoT training using the appropriate YAML file (e.g., `multimodal_cot.yaml`).
    *   Select the best checkpoint based on validation performance.

3.  **Multimodal CoCoNuT Training:**
    *   Configure the CoCoNuT training in `args/multimodal_coconut.yaml`.
    *   Set `load_model_path` to the path of your best M-CoT checkpoint.
    *   Run the training script. The `run.py` script will automatically handle the staged curriculum.

4.  **Evaluation:**
    *   Use the evaluation configuration file.
    *   Set `load_model_path` to your best Multimodal CoCoNuT checkpoint and set `only_eval: True`.
    *   Run the script to report final performance on the test set.

## 8. Conclusion

Multimodal CoCoNuT extends the innovative reasoning paradigm of CoCoNuT into the vision-language domain. By enabling models to reason in a continuous, multimodal latent space, it opens up new possibilities for more powerful, flexible, and transparent reasoning on complex multimodal tasks. The staged training curriculum provides a robust method for teaching models this complex skill, building upon the strong foundation of pre-trained vision-language models like InternVL3.

## 9. Detailed Codebase Explanation

### 9.1. `run.py` - The Orchestrator

This script is the main entry point for training and evaluation. It performs the following key functions:

*   **Configuration Loading:** Loads the experiment configuration from a YAML file (e.g., `args/multimodal_coconut.yaml`) into a `Config` object.
*   **Distributed Training Setup:** Initializes the distributed training environment using `torch.distributed` and sets up either DDP or FSDP based on the configuration.
*   **Model and Tokenizer Initialization:** Loads the InternVL3 base model and tokenizer, adding the special tokens required for CoCoNuT (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`).
*   **Staged Training Loop:** Manages the main training loop, iterating through epochs and calculating the `scheduled_stage` for each epoch based on `epoch // epochs_per_stage`.
*   **Data Loading:**  Calls the functions in `multimodal_coconut/data/dataset.py` to prepare the appropriate datasets for the current stage of the training curriculum.
*   **Trainer Initialization:** Initializes the `MultimodalCotTrainer` which handles the actual training and validation steps.
*   **Checkpointing:** Saves model checkpoints at specified intervals.

### 9.2. The Data Pipeline: `multimodal_coconut/data/`

The data pipeline is responsible for loading, preprocessing, and batching the multimodal data.

*   **`dataset.py`:**
    *   **`MultimodalDataset`:** This class inherits from `torch.utils.data.Dataset`. It loads the A-OKVQA dataset from a JSON file, and for each sample, it loads the corresponding image, preprocesses it using the `ImageProcessor`, and tokenizes the question, reasoning steps, and answer.
    *   **`MultimodalCollator`:** This class is responsible for creating batches of data. It pads the text sequences to the same length and stacks the image tensors. Crucially, it also aligns the latent tokens in the batch to optimize the training process.

*   **`image_processor.py`:**
    *   **`ImageProcessor`:** This class wraps InternVL3's image preprocessing pipeline. It handles image resizing, normalization, and tiling, preparing the images to be fed into the model's vision encoder.

*   **`prepare_aokvqa.py`:**
    *   This script contains functions to download and preprocess the A-OKVQA dataset, converting it into the JSON format expected by the `MultimodalDataset`.

### 9.3. The Model: `multimodal_coconut/model/`

This is where the core logic of the Multimodal CoCoNuT model resides.

*   **`multimodal_coconut.py`:**
    *   **`MultimodalCoconut`:** This `nn.Module` class is the heart of the project. It wraps the base InternVL3 model.
        *   **`__init__`:** Initializes the model, including the base model and the special token IDs.
        *   **`forward`:** This method contains the core CoCoNuT logic, adapted for multimodal inputs. It takes `pixel_values` (the preprocessed images) and `input_ids` (the tokenized text) as input. It identifies the locations of the `<|latent|>` tokens and processes the sequence iteratively. When it encounters a latent token, it uses the hidden state of the *previous* token as the input embedding for the current latent token. This creates the continuous thought feedback loop. The `kv_cache` is used to make this multi-pass process efficient.
        *   **`generate`:** This method handles inference. It generates a sequence of tokens, which can include both discrete text and continuous thought steps, conditioned on the input image and question.

### 9.4. The Training System: `multimodal_coconut/training/`

This module contains the logic for training the model.

*   **`multimodal_cot_trainer.py`:**
    *   **`MultimodalCotTrainer`:** This class orchestrates the training process for a single epoch. It contains the `train_epoch` method, which iterates over the training dataloader, performs the forward and backward passes, and updates the model's weights. It also includes a `validate` method to evaluate the model on the validation set.

*   **`stage_manager.py`:**
    *   **`StageManager`:** This class manages the staged training curriculum. It determines the current stage based on the epoch number and prepares the data accordingly by calling the appropriate functions in the `dataset.py` module.

### 9.5. Configuration: `multimodal_coconut/config/`

*   **`config.py`:**
    *   **`Config`:** A simple class that loads the YAML configuration file and provides access to the hyperparameters as attributes. This allows for easy and flexible experiment configuration.

### 9.6. Utilities: `multimodal_coconut/utils/`

This module contains various utility functions.

*   **`distributed.py`:** Contains functions for setting up and managing distributed training.
*   **`logging.py`:** Provides logging utilities for tracking experiment progress.
*   **`misc.py`:** Contains other miscellaneous helper functions.