# Fine-Tuning Script Documentation

## Part 1: Project Description

The primary goal of this project is to fine-tune the `stable-code-3b` model, a transformer language model with about 2.7 billion parameters, designed for enhanced code comprehension and generation across a variety of programming languages.

A key challenge of this project is to adapt the model to efficiently process new data while maintaining optimal performance. The project leverages a dataset of extensive, open-source code, carefully filtered and prepared for this task. A major objective is to improve the code generation capabilities by adjusting the model's weights using tailored fine-tuning scripts.

A significant technical challenge is performing the fine-tuning using a GeForce RTX 3080 GPU, with its 10GB of VRAM, to demonstrate that advanced model fine-tuning can be performed locally without cloud infrastructure. This necessitates innovative strategies to handle the model's size within limited GPU memory.

## Part 2: Function Explanation

```python
import pandas as pd
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from accelerate import Accelerator
import numpy as np
import random
from IPython.display import display
import evaluate
import logging
from datasets import Dataset

import warnings
warnings.filterwarnings("ignore")
```

- Essential libraries and settings initialization.

### `compute_metrics(eval_pred)`

This function computes the accuracy metric for evaluating model performance:

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]
    return accuracy_metric.compute(predictions=predictions, references=labels)
```

- **Line 1-2**: Receives the model's predicted logits and true labels.
- **Line 3**: Uses `numpy` to determine the class with the highest score for each prediction.
- **Line 4-5**: Flattens the arrays for seamless processing.
- **Line 6-7**: Filters out padding indices (`-100` by convention) to consider only valid token predictions.
- **Line 8**: Computes accuracy by comparing predictions against the true labels.

### `load_model(model_path)`

This function loads a pre-trained model and tokenizer, optimized for GPU efficiency:

```python
def load_model(model_path):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{6000}MB'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype="auto",
        device_map="cuda:0",
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
```

- **Line 2**: Determines and logs the count of available GPUs for resource management.
- **Line 3**: Configures the maximum allowable memory allocated to each GPU.
- **Line 4-9**: Loads the language model in 4-bit quantization to save GPU memory space.
- **Line 10**: Loads the tokenizer for tokenizing input data.
- **Line 11**: Sets the end-of-sequence token as the padding token to ensure consistency.
- **Line 12**: Returns the initialized model and tokenizer for further tasks.

### `create_prompt_formats(sample)`

This function organizes sample data into a structured prompt format:

```python
def create_prompt_formats(sample):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample
```

- **Line 2-6**: Defines constant strings that represent different sections of the prompt.
- **Line 8-12**: Constructs each part of the prompt, combining static text with dynamic data.
- **Line 13**: Collects non-null parts into a list.
- **Line 14**: Joins all parts into one formatted string with double newlines separating sections.
- **Line 15**: Attaches the formatted string to the sample as the `text` field.
- **Line 16**: Returns the modified sample containing the newly formatted prompt.

### `get_max_length(model)`

Fetches the maximum sequence length the model can process:

```python
def get_max_length(model):
    conf = model.config
    max_length = getattr(conf, 'n_positions', None) or getattr(conf, 'max_position_embeddings', None) or getattr(conf, 'seq_length', None) or 1024
    print(max_length)
    return max_length
```

- **Line 2**: Accesses the model's configuration.
- **Line 3**: Attempts to retrieve known configuration attributes for maximum length, defaulting to 1024 if unspecified.
- **Line 4**: Logs the determined maximum length.
- **Line 5**: Returns this maximum length for tokenizer use.

### `preprocess_batch(batch, tokenizer, max_length)`

This function tokenizes text batches, conforming to model’s maximum input length:

```python
def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)
```

- **Line 2**: Employs the tokenizer on input batch texts, ensuring they do not exceed the configured `max_length`.

### `preprocess_dataset(tokenizer, max_length, seed, dataset)`

Formats and tokenizes the entire dataset, aligning data for training:

```python
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    dataset = dataset.apply(create_prompt_formats, axis=1)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset['input_ids'] = dataset['text'].apply(lambda x: _preprocessing_function({'text': x})['input_ids'])
    dataset = dataset[dataset['input_ids'].apply(len) < max_length]
    dataset = dataset.sample(frac=1, random_state=seed)  # Shuffle the dataset
    return dataset
```

- **Line 2**: Prints a message to indicate the start of preprocessing.
- **Line 3**: Applies prompt formatting to each dataset entry.
- **Line 4**: Prepares a partial function for batch tokenization.
- **Line 5**: Tokenizes all sample texts, storing results in the `input_ids`.
- **Line 6**: Filters examples longer than the allowable token sequence.
- **Line 7**: Shuffles the dataset using a random seed for reproducibility.
- **Line 8**: Outputs the preprocessed dataset ready for training.

### `create_peft_config(modules)`

Sets up a configuration for Parameter-Efficient Fine-Tuning (PEFT) via LoRA:

```python
def create_peft_config(modules):
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",    # the choice of the task is conditioning the loss function used later in Trainer class, here it will be cross-entropy due to the nature of the task
    )
```

- **Line 2-8**: Specifies the rank (`r`), scaling factor (`alpha`), dropout, targeted modules, and task type, which collectively guide the fine-tuning process.

### `find_all_linear_names(model)`

Finds all linear layers in the model suitable for LoRA modifications:

```python
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
```

- **Line 2**: Initializes an empty set for storing the names of linear layers.
- **Line 3-5**: Iterates over model components, selecting and logging those that are linear.
- **Line 6-7**: Removes `lm_head` if present; it is intentionally not modified.
- **Line 8**: Converts and returns the set as a list of linear component names.

### `print_trainable_parameters(model)`

Logs the number of total and trainable parameters within the model:

```python
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param:.2f}")
```

- **Line 2**: Sets up counters for parameters.
- **Line 3-7**: Totals all parameters and increments those marked for training.
- **Line 8**: Outputs a summary indicating the volume and ratio of trainable parameters.

### `train_with_accelerate(model, tokenizer, train_dataset, eval_dataset, output_dir)`

Oversees the data preprocessing, model setup, and execution of training through `Accelerator`:

```python
def train_with_accelerate(model, tokenizer, train_dataset, eval_dataset, output_dir):
    logger.info("Starting training")

    accelerator = Accelerator(device_placement=True)               # Enable offloading to CPU and RAM if needed
    model, train_dataset, eval_dataset = accelerator.prepare(
        model, train_dataset, eval_dataset
    )

    # Prepare model for k-bit training and locate linear modules
    model = prepare_model_for_kbit_training(model)

    # Freeze all layers except the last one
    for name, param in model.named_parameters():
        if "lm_head" not in name:  # Change "lm_head" according to the name of the final layer in your model
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Output the trainable parameters
    print_trainable_parameters(model)
    
    # Obtain PEFT model
    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    model = get_peft_model(model, peft_config)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4, 
            warmup_steps=20,
            max_steps=2000,
            learning_rate=2e-5,
            logging_steps=2,
            output_dir="outputs", 
            optim="paged_adamw_8bit",
            weight_decay=0.01,    # regularization, penalty linked to weights amplitude
            eval_strategy="no",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False

    # Training
    train_result = trainer.train()
    logger.info(f"Training completed. Metrics: {train_result.metrics}")

    # Evaluate
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation completed. Results: {eval_result}")

    # Save trained model
    trainer.model.save_pretrained(output_dir)

    # Clear resources
    del model
    del trainer
    torch.cuda.empty_cache()
```

- **Line 2**: Logs that the training process is beginning.
- **Line 4**: Initializes the `Accelerator`, preparing it for efficient GPU/CPU management.
- **Line 5-6**: Adjusts the model and datasets for accelerated execution, potentially offloading tasks.
- **Line 9**: Optimizes the model for reduced precision, enabling layer-wise fine-tuning.
- **Line 12-16**: Ensures only final layers are trainable, maintaining frozen states for others.
- **Line 19**: Outputs a summary of which parameters are trainable.
- **Line 22-27**: Identifies linear layers suitable for adaptation and configures LoRA settings.
- **Line 30-43**: Configures the `Trainer`, defining input arguments for batch sizes, learning rates, logging, and evaluation strategies.
- **Line 46**: Disables caching to conserve memory during training.
- **Line 49-51**: Executes training and logs resulting metrics.
- **Line 54-56**: Executes model evaluation, logging results.
- **Line 59**: Saves the trained model state for future use.
- **Line 62-64**: Frees allocated memory and resources to prevent leaks post-training.

### Training Process Initialization

Within the entry-point script, these segments establish configuration, load the dataset, and initiate the fine-tuning endeavor using the aforementioned functions. By the end of this run, a fine-tuned model is saved, ready for deployment or further testing.

Overall, these functions collectively achieve efficient training and fine-tuning of the `stable-code-3b` model on resource-constrained hardware by leveraging techniques such as parameter-efficient fine-tuning, careful resource allocation via `Accelerator`, and comprehensive pre-processing workflows.

## Part 3: Processes and Choices Explanation

To ensure efficient fine-tuning, several strategic choices were made:

- **LoRA Technique**: Low-Rank Adaptation was employed to reduce the model's computational load by focusing parameter updates on select modules, accommodating hardware constraints.

- **Quantization and Memory Management**: Utilizing 4-bit quantization significantly reduced memory load, enabling operation within limited VRAM capacities of the available hardware.

- **Accelerator Usage**: Offloading computational tasks appropriately using the `Accelerator` library provided better resource management and efficiency during training.

- **Freezing Layers Technique**: Freezing non-essential layers curtailed the total trainable parameters, aligning the process with available resources.

- **Optimization Strategies**: Configuration settings were optimized to address VRAM fragmentation and improve overall memory utilization.

These approaches collectively enabled successful local fine-tuning of a large-scale model on constrained resources, demonstrating practical strategies for bringing advanced AI solutions to environments with limited computational capabilities.