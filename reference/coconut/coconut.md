# CoCoNuT: Reasoning in a Continuous Latent Space

This document provides a detailed explanation of the CoCoNuT (Chain of Continuous Thought) methodology, a novel approach for training Large Language Models (LLMs) to reason in a continuous latent space. We will delve into the core concepts, the training algorithms for both traditional Chain-of-Thought (CoT) and CoCoNuT, and provide a deep dive into the codebase.

Research paper : https://arxiv.org/abs/2412.06769

## 1. Core Concepts

The fundamental difference between CoT and CoCoNuT lies in how they represent and process intermediate reasoning steps.

### Chain-of-Thought (CoT)

In the standard CoT paradigm, the model generates explicit reasoning steps as discrete text tokens. The output of the model at each step is a token, which is then fed back as an input for the next step. This process is sequential and autoregressive, essentially creating a "chain" of textual thoughts that lead to the final answer.

### Chain of Continuous Thought (CoCoNuT)

CoCoNuT replaces these discrete textual reasoning steps with "continuous thoughts." Instead of generating a token, the model's internal state (specifically, the last hidden state) is used as a representation of the thought. This continuous vector is then directly fed back into the model as the input embedding for the next reasoning step. This allows for a more fluid and potentially more expressive way of reasoning, akin to a breadth-first search over the latent space of possibilities.

The following diagram from the research paper visually contrasts the two approaches:

![CoT vs. CoCoNuT](https://github.com/facebookresearch/coconut/raw/main/assets/coconut.png)

## 2. Training Methodology

The training process is staged, starting with a CoT-trained model and then progressively introducing continuous thoughts.

### 2.1. CoT Training (Stage 0)

The foundation of a CoCoNuT model is a model first trained with standard Chain-of-Thought.

**Algorithm:**

1.  **Data Preparation:** The training data consists of `(question, steps, answer)` triplets. The `steps` are a list of textual reasoning steps.
2.  **Input Formatting:** The input to the model is a concatenation of the question and the ground-truth reasoning steps.
3.  **Training Objective:** The model is trained to predict the next token in the sequence, including both the reasoning steps and the final answer. The loss function is a standard cross-entropy loss over the vocabulary.
4.  **Configuration:** In the codebase, this is controlled by the `args/gsm_cot.yaml` file, where `cot: True` and `coconut: False`.

This initial training teaches the model the fundamental reasoning patterns of the task.

### 2.3. The CoCoNuT Training Curriculum: From CoT to Continuous Thought

The transition from a standard CoT model to a CoCoNuT model is not a single step but a carefully designed curriculum. This staged approach is crucial for the model's ability to learn to reason in the continuous latent space.

**The Core Idea:** The curriculum gradually weans the model off of explicit textual reasoning steps and encourages it to represent those steps in its own internal, continuous space. This is achieved by progressively replacing the ground-truth textual steps with `<|latent|>` tokens during training.

**The Stages of Training:**

The training process is divided into "stages," with each stage lasting for a certain number of epochs (controlled by `epochs_per_stage` in the `.yaml` config). The current stage is determined by the `scheduled_stage` variable in `run.py`, which is calculated as `epoch // epochs_per_stage`.

*   **Stage 0 (CoT Pre-training):**
    *   **Goal:** Teach the model the basic reasoning patterns of the task using standard Chain-of-Thought.
    *   **Data:** The model is fed the `(question, steps, answer)` data, where `steps` are the complete, ground-truth textual reasoning steps.
    *   **Implementation:** This is either done as a separate pre-training run (as in the GSM8K experiments) or as the initial stage of the CoCoNuT training.

*   **Stage 1:**
    *   **Goal:** Introduce the concept of a single continuous thought.
    *   **Data:** The *first* textual reasoning step is replaced by `c_thought` number of `<|latent|>` tokens. The model is then expected to generate the *rest* of the reasoning steps and the final answer.
    *   **Example:**
        *   **Original Input:** `[Question] [Step 1] [Step 2] [Step 3] [Answer]`
        *   **Stage 1 Input:** `[Question] <|latent|> <|latent|> [Step 2] [Step 3] [Answer]` (assuming `c_thought=2`)
    *   **Learning Objective:** The model must learn to encode the information that would have been in "Step 1" into the continuous thought vectors produced at the latent token positions. This latent representation must then be sufficient for the model to correctly generate "Step 2" and the subsequent steps.

*   **Stage 2 and Beyond:**
    *   **Goal:** Gradually increase the number of continuous thoughts, forcing the model to rely more on its internal reasoning process.
    *   **Data:** In each subsequent stage, one more textual reasoning step is replaced by `<|latent|>` tokens.
    *   **Example (Stage 2):**
        *   **Input:** `[Question] <|latent|> <|latent|> <|latent|> <|latent|> [Step 3] [Answer]`
    *   **Progressive Deepening:** This process continues up to `max_latent_stage`. By the final stage, the model is expected to perform the first `max_latent_stage` reasoning steps entirely in the continuous latent space, only decoding to text for the final few steps and the answer.

**The "Aha!" Moment - Why this works:**

The staged curriculum provides a stable learning signal for the model. It's not expected to learn the entire complex process of continuous reasoning at once. Instead, it builds on its existing knowledge from the previous stage.

*   In Stage 1, it has the strong anchor of the ground-truth "Step 2" to guide its learning of the first continuous thought.
*   In Stage 2, it has the anchor of "Step 3" to guide its learning of the second continuous thought, and so on.

This gradual process allows the model to develop a robust internal representation of the reasoning process, moving beyond simple token-matching to a more abstract, continuous understanding of the problem-solving steps. The `uniform_prob` parameter in the configuration allows for mixing data from different stages, which can help to improve the model's generalization and prevent it from "forgetting" how to handle shorter reasoning chains.

This curriculum is the key to unlocking the power of CoCoNuT, enabling it to perform the more flexible and powerful reasoning observed in the research paper.

Once a base CoT model is trained, the CoCoNuT training begins. This is a staged process where the model is gradually taught to use continuous thoughts.

**Algorithm:**

1.  **Model Initialization:** The CoCoNuT model is initialized with the weights of the pre-trained CoT model. This is specified in the `args/gsm_coconut.yaml` file with the `load_model_path` parameter.
2.  **Staged Training:** The training is divided into stages. In each stage, a certain number of initial textual reasoning steps are replaced by special `<|latent|>` tokens.
    *   The `run.py` script manages this staged training loop. The `scheduled_stage` variable, calculated as `epoch // epochs_per_stage`, determines the number of reasoning steps to replace.
3.  **Data Preparation for CoCoNuT:** The `get_cot_latent_dataset` function in `dataset.py` is responsible for creating the training data for each stage. It takes the original `(question, steps, answer)` data and, based on the `scheduled_stage`, replaces the initial `k` steps with `k * c_thought` latent tokens.
4.  **Continuous Thought Feedback:** The core of the CoCoNuT mechanism is implemented in the `forward` pass of the `Coconut` class in `coconut.py`.
    *   When the model encounters a `<|latent|>` token, it does not predict a new token from the vocabulary.
    *   Instead, it takes the last hidden state from the *previous* token and uses that hidden state vector as the input embedding for the current latent token.
    *   This creates a direct feedback loop of the model's internal state, allowing it to reason in the continuous latent space.
5.  **Progressive Deepening:** As the training progresses through stages, more and more textual reasoning steps are replaced by latent tokens, forcing the model to rely increasingly on its continuous thought process.

## 3. Codebase Deep-Dive

This section provides an overview of the key files in the `reference/coconut` directory.

### `run.py` - The Orchestrator

This is the main script for both training and evaluation.

*   **Responsibilities:**
    *   Parses the command-line arguments, including the path to the configuration file.
    *   Initializes the distributed training environment (DDP/FSDP).
    *   Loads the base model and tokenizer, adding the special tokens required for CoCoNuT (`<|start-latent|>`, `<|end-latent|>`, `<|latent|>`).
    *   Manages the staged training loop, calculating the `scheduled_stage` for each epoch.
    *   Calls the functions in `dataset.py` to prepare the appropriate datasets for the current stage.
    *   Handles the training loop, backpropagation, and optimizer steps.
    *   Performs validation at the end of each epoch.
    *   Saves model checkpoints.

### `coconut.py` - The CoCoNuT Model

This file contains the core implementation of the CoCoNuT model.

*   **`Coconut` Class:**
    *   An `nn.Module` that wraps a base causal language model (e.g., GPT-2).
    *   The `__init__` method takes the base model and the special token IDs as input.
    *   **`forward` method:** This is where the magic happens.
        1.  It identifies the locations of all `<|latent|>` tokens in the input batch.
        2.  It processes the input sequence in segments. When it reaches a latent token, it performs a forward pass up to that point.
        3.  It then takes the `hidden_states` from the last layer of the base model.
        4.  The hidden state of the token *preceding* the latent token is then used as the input embedding for the latent token itself.
        5.  This process is repeated for all latent tokens in the sequence, effectively chaining the continuous thoughts.
        6.  It uses a `kv_cache` to make this multi-pass process more efficient.
    *   **`generate` method:** Handles inference, generating a sequence of tokens that may involve both discrete and continuous thought steps.

### `dataset.py` - Data Loading and Preparation

This script is responsible for preparing the data in the format required for training and evaluation.

*   **`get_dataset`:** Loads the raw JSON data and tokenizes the questions, steps, and answers.
*   **`MyCollator`:** A custom data collator that intelligently pads batches containing latent tokens. This is crucial for efficiency, as it aligns the latent tokens to maximize the reuse of the KV cache during the forward pass.
*   **`get_cot_latent_dataset`:** Prepares the training data for a given stage. It replaces the initial `k` reasoning steps with `<|latent|>` tokens.
*   **`get_question_latent_dataset`:** Prepares data for validation and testing by inserting the appropriate number of latent tokens based on the current stage.

### `args/*.yaml` - Configuration System

The `args` directory contains YAML files that define the hyperparameters and settings for different experiments.

*   **`gsm_cot.yaml`:** Configuration for training the base CoT model.
*   **`gsm_coconut.yaml`:** Configuration for training the CoCoNuT model. Key parameters include:
    *   `coconut: True`: Enables the CoCoNuT model.
    *   `c_thought`: The number of continuous thoughts per reasoning step.
    *   `epochs_per_stage`: The number of epochs in each training stage.
    *   `max_latent_stage`: The maximum number of reasoning steps to replace with latent thoughts.
    *   `load_model_path`: Path to the pre-trained CoT model checkpoint.

## 4. Algorithms and Function Descriptions

This section provides a more detailed look at the algorithms and the purpose of key functions in the training process, with a focus on the reasoning behind the implementation choices.

### 4.1. CoT Training

The initial stage of the training process is to establish a strong baseline model that understands the task and the structure of the reasoning process. This is done through standard Chain-of-Thought (CoT) fine-tuning.

**Algorithm:**

The CoT training algorithm is a standard auto-regressive language modeling task.

```
1. For each (question, steps, answer) in the training data:
2.   Concatenate and tokenize the input: 
     `input_ids = tokenize(question + "\n" + "\n".join(steps) + "\n### " + answer)`
3.   The `labels` are a copy of `input_ids`, as the model is trained to predict the next token at every position.
4.   For each training epoch:
5.     For each batch:
6.       Perform a forward pass: `outputs = model(input_ids, labels=labels)`
7.       Calculate the cross-entropy loss: `loss = outputs.loss`
8.       Perform a backward pass: `loss.backward()`
9.       Update the model's weights: `optimizer.step()`
```

**Key Functions and Reasoning:**

*   `run.py:main()`: This function acts as the main entry point. For CoT training, it ensures that the `coconut` flag is `False` and the `cot` flag is `True` in the configuration. It sets up a standard training loop without any of the special staged logic required for CoCoNuT.
*   `dataset.py:get_dataset()`: This function is responsible for the initial data ingestion and tokenization. It takes the raw JSON data and converts the `question`, `steps`, and `answer` fields into token IDs. The key here is that it preserves the full sequence of reasoning steps, which is essential for the CoT training objective.
*   `dataset.py:get_cot_latent_dataset()`: In the context of CoT training (`scheduled_stage = 0`), this function's role is to simply format the data without introducing any latent tokens. It concatenates the tokenized question, all the reasoning steps, and the final answer into a single sequence. This creates the ground-truth data that the model will be trained on.

### 4.2. CoCoNuT Training

This is where the core innovation of the paper is implemented. The training process is designed to gradually shift the model's reasoning from the discrete token space to the continuous latent space.

**Algorithm:**

The CoCoNuT training algorithm is a staged process that progressively replaces textual reasoning steps with continuous thought vectors.

```
1. Load the pre-trained CoT model as the starting point.
2. For each training epoch:
3.   Determine the current stage: `scheduled_stage = epoch // epochs_per_stage`
4.   For each (question, steps, answer) in the training data:
5.     Determine the number of steps to replace: `k = min(scheduled_stage, max_latent_stage)`
6.     Calculate the number of latent tokens to insert: `num_latent = k * c_thought`
7.     Prepare the input sequence:
       `input_ids = tokenize(question) + [latent_id] * num_latent + tokenize(steps[k:] + answer)`
8.     The `labels` are created to only supervise the generation of the remaining textual steps and the final answer. The latent token positions are ignored in the loss calculation (typically by setting their label to -100).
9.   For each batch:
10.    **Forward Pass (The `Coconut.forward` method):**
11.      The input sequence is processed token by token.
12.      When a non-latent token is encountered, a standard transformer forward pass is performed.
13.      When a `latent_id` is encountered at position `i`:
14.        a. The forward pass is paused.
15.        b. The hidden state vector `h` from the output of the *previous* position (`i-1`) is retrieved.
16.        c. This hidden state `h` is then used as the input embedding for the current position `i`. This is the "continuous thought" feedback loop.
17.        d. The forward pass resumes from position `i`.
18.    Calculate the loss only on the parts of the sequence that correspond to the remaining textual steps and the answer.
19.    Perform a backward pass and update the model's weights.
```

**Key Functions and Reasoning:**

*   `run.py:main()`: The orchestrator of the training curriculum. It calculates the `scheduled_stage` at the beginning of each epoch, effectively controlling how many reasoning steps are replaced by latent tokens. This gradual increase in `scheduled_stage` is the core of the curriculum.
*   `dataset.py:get_cot_latent_dataset()`: This function is the heart of the data preparation for the curriculum. Based on the `scheduled_stage`, it dynamically creates the input sequences for the model. Its primary responsibility is to replace the initial `k` textual reasoning steps with the appropriate number of `<|latent|>` tokens, setting up the learning problem for the current stage.
*   `coconut.py:Coconut.forward()`: This is the most critical part of the implementation. It deviates from the standard forward pass of a language model. The multi-pass approach with the `kv_cache` is a clever optimization to handle the feedback loop efficiently. By iteratively processing the sequence and feeding back the hidden states, it allows the model to build a chain of continuous thoughts. The logic to identify latent tokens and substitute their embeddings with the previous hidden state is the direct implementation of the CoCoNuT mechanism described in the paper.
*   `coconut.py:Coconut.generate()`: During inference, this function mirrors the logic of the `forward` pass. When it needs to generate a "thought," it doesn't sample a token from the vocabulary. Instead, it performs the same continuous thought feedback, allowing the model to reason in the latent space before generating the final textual answer. This is why CoCoNuT can be more efficient at inference time, as it can perform multiple reasoning "steps" in the latent space without the overhead of generating and re-encoding text.

## 5. Critical Implementation Details and 'Aha!' Moments

To truly grasp the CoCoNuT implementation, we need to look beyond the high-level algorithms and examine the specific coding patterns that make it work. These details are critical for anyone looking to understand, replicate, or build upon this research.

### The Multi-Pass `forward` Method: The Engine of Continuous Thought

The core of CoCoNuT's magic lies in the `forward` method of the `Coconut` class. A standard language model performs a single forward pass over the entire input sequence. CoCoNuT, however, cannot do this.

*   **Why a Single Pass is Impossible:** The input embedding for a latent token at position `i` *depends on the output* (the hidden state) of the model at position `i-1`. This creates a sequential dependency that cannot be resolved in a single, parallelized forward pass.

*   **The Solution: Iterative Forward Passes:** The code solves this by performing a series of smaller forward passes.
    1.  It first finds the position of the *first* latent token in the batch.
    2.  It performs a forward pass on all the tokens *up to* that point.
    3.  It then uses the resulting hidden state to construct the input for the first latent token.
    4.  It then performs another forward pass for that single latent token.
    5.  This process repeats, one latent token at a time, until all continuous thoughts have been processed.
    6.  Finally, a last forward pass is done on the remaining textual part of the sequence.

*   **The 'Aha!' Moment:** This iterative process is the direct implementation of the "chain" in "Chain of Continuous Thought." Each pass is a link in the chain, and the information is passed not as a discrete token, but as a high-dimensional vector (the hidden state).

### The `kv_cache`: Making the Multi-Pass Efficient

Performing a full forward pass for each latent token would be incredibly inefficient. The `kv_cache` is the key optimization that makes this feasible.

*   **How it Works:** In a transformer, the Key (K) and Value (V) matrices are calculated for each token and are used in the attention mechanism. The `kv_cache` stores these matrices.
*   **The Benefit:** When the model performs the forward pass for the second latent token, it doesn't need to re-calculate the K and V matrices for all the preceding tokens. It can simply reuse them from the cache.
*   **The 'Aha!' Moment:** The `kv_cache` transforms the computationally expensive multi-pass process into something much more manageable. It ensures that the model only performs the "new" computations required for the current token, while reusing the context from all the previous tokens.

### The Role of Special Tokens and Hyperparameters

*   **`<|start-latent|>` and `<|end-latent|>`:** While the paper focuses on the `<|latent|>` token, the start and end markers serve as important delimiters. They provide the model with clear signals about where the continuous thought sequence begins and ends, which can help to stabilize the learning process. In the current implementation, they are not strictly necessary for the core logic but are good practice for more complex future extensions.
*   **`c_thought`:** This hyperparameter controls the "length" of a continuous thought. A value of `2` means that each textual reasoning step is replaced by *two* latent tokens. This gives the model more "time" or "space" in the latent dimension to perform the reasoning that would have been done in a single textual step. The 'Aha!' moment here is that a single discrete step in text might correspond to multiple, finer-grained computational steps in the latent space.

### `MyCollator`: The Unsung Hero of Batching

The `MyCollator` class in `dataset.py` is another critical piece of the puzzle. When batching sequences of different lengths, the standard approach is to pad them all to the length of the longest sequence. However, CoCoNuT has a unique requirement.

*   **The Challenge:** To maximize the benefit of the `kv_cache`, we want the latent tokens to be as aligned as possible across the different examples in a batch.
*   **The Solution:** The `MyCollator` intelligently adds padding *at the beginning* of the sequences. It finds the position of the earliest latent token in the batch and adds leading padding to all other sequences to align them.
*   **The 'Aha!' Moment:** This custom padding strategy is a performance optimization that is not immediately obvious. It ensures that the initial, non-latent part of the sequences can be processed in a single, efficient forward pass, maximizing the reuse of the `kv_cache` before the iterative latent token processing begins.

## 6. Practical Replication Guide

This guide provides a step-by-step walkthrough for replicating the CoCoNuT experiments.

### Step 1: Environment and Data Setup

1.  **Clone the repository and set up the environment:**
    ```bash
    git clone git@github.com:facebookresearch/coconut.git
    cd coconut
    conda create --name coconut python=3.12
    conda activate coconut
    pip install -r requirements.txt
    ```
2.  **Prepare the dataset:** For GSM8K, run the provided script. This will download the data and format it into the required JSON structure.
    ```bash
    bash preprocessing/gsm_icot.bash
    ```

### Step 2: CoT Pre-training (Stage 0)

The first step is to train a baseline CoT model. This model will serve as the foundation for the CoCoNuT training.

1.  **Configure the CoT training:** Open `args/gsm_cot.yaml`. Ensure the settings are correct for your setup (e.g., `model_id`, `batch_size_training`).
2.  **Run the CoT training:**
    ```bash
    torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/gsm_cot.yaml
    ```
3.  **Select the best checkpoint:** After the training is complete, you will have a series of checkpoints in the `checkpoints/gsm-cot` directory. Monitor the validation accuracy during training and select the checkpoint that achieved the best performance. This will be the `load_model_path` for the next stage.

### Step 3: CoCoNuT Training (The Curriculum)

Now, you will use the CoT model as a starting point and train it using the staged CoCoNuT curriculum.

1.  **Configure the CoCoNuT training:** Open `args/gsm_coconut.yaml`.
    *   Set `load_model_path` to the path of the best CoT checkpoint you selected in the previous step.
    *   Adjust other parameters like `c_thought`, `epochs_per_stage`, and `max_latent_stage` as needed for your experiment.
2.  **Run the CoCoNuT training:**
    ```bash
    torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/gsm_coconut.yaml
    ```
    This will start the staged training process. The `run.py` script will automatically handle the curriculum, increasing the number of latent tokens as the epochs progress.

### Step 4: Evaluation

Once the CoCoNuT training is complete, you can evaluate its performance on the test set.

1.  **Configure the evaluation:** Open `args/gsm_coconut_eval.yaml`.
    *   Set `load_model_path` to the path of the best CoCoNuT checkpoint from the previous step.
    *   Ensure `only_eval` is set to `True`.
2.  **Run the evaluation:**
    ```bash
    torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/gsm_coconut_eval.yaml
    ```
    This will run the model on the test set and report the final accuracy.

By following these steps, you can faithfully replicate the training and evaluation process described in the research paper.

## 7. Conclusion

CoCoNuT introduces a paradigm shift in how we think about reasoning in LLMs. By moving from discrete, textual reasoning to a continuous latent space, it opens up new possibilities for more powerful and flexible reasoning. The staged training methodology is a key innovation, allowing the model to gradually learn this complex new skill. The codebase provides a clear and well-structured implementation of these ideas, offering a solid foundation for future research in this exciting area.
