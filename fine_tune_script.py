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
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner logs

#_______________________________________________________________________

# Function to compute metrics for model evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Separate logits and true labels
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predicted labels
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Ignore padding indices (-100) when computing accuracy
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    # Compute and return accuracy
    return accuracy_metric.compute(predictions=predictions, references=labels)

#_______________________________________________________________________

# Function to load a pre-trained model and its tokenizer
def load_model(model_path):
    n_gpus = torch.cuda.device_count()  # Count available GPUs
    max_memory = f'{6000}MB'  # Define maximum memory allocation per GPU

    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,  # Use quantization for efficient inference
        torch_dtype="auto",
        device_map="cuda:0",  # Map the model to the first CUDA device
        max_memory={i: max_memory for i in range(n_gpus)},
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding token

    return model, tokenizer

#_______________________________________________________________________

# Function to create formatted prompts for each dataset sample
def create_prompt_formats(sample):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "### Input:"
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"

    # Construct the formatted prompt
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    # Combine the parts, excluding None values
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample

#_______________________________________________________________________

# Function to retrieve the maximum sequence length supported by the model
def get_max_length(model):
    conf = model.config
    max_length = getattr(conf, 'n_positions', None) or getattr(conf, 'max_position_embeddings', None) or getattr(conf, 'seq_length', None) or 1024
    print(max_length)
    return max_length

#_______________________________________________________________________

# Function to preprocess a single batch of text using the tokenizer
def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)

#_______________________________________________________________________

# Function to preprocess an entire dataset
def preprocess_dataset(tokenizer, max_length, seed, dataset):
    print("Preprocessing dataset...")
    dataset = dataset.apply(create_prompt_formats, axis=1)  # Apply formatting to each sample

    # Partial function for preprocessing batches
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    # Tokenize and filter samples exceeding the max length
    dataset['input_ids'] = dataset['text'].apply(lambda x: _preprocessing_function({'text': x})['input_ids'])
    dataset = dataset[dataset['input_ids'].apply(len) < max_length]

    # Shuffle the dataset
    dataset = dataset.sample(frac=1, random_state=seed)
    return dataset

#_______________________________________________________________________

# Function to create a configuration for parameter-efficient fine-tuning (PEFT)
def create_peft_config(modules):
    return LoraConfig(
        r=8,  # Rank of the decomposition
        lora_alpha=32,  # Scaling factor
        target_modules=modules,  # Modules to apply LoRA
        lora_dropout=0.05,  # Dropout rate
        bias="none",  # Bias handling
        task_type="CAUSAL_LM",  # Task type
    )

#_______________________________________________________________________

# Function to find all linear module names in the model for LoRA application
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')  # Exclude the output head
    return list(lora_module_names)

#_______________________________________________________________________

# Function to print trainable vs total parameters in the model
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}")

#_______________________________________________________________________

# Function to train the model using the `Accelerate` library
def train_with_accelerate(model, tokenizer, train_dataset, eval_dataset, output_dir):
    logger.info("Starting training")

    accelerator = Accelerator(device_placement=True)  # Enable offloading to CPU and RAM if needed
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
            weight_decay=0.01,    # Regularization term
            eval_strategy="no",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False  # Disable caching for training

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

#_______________________________________________________________________

if __name__ == "__main__":

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Configure PyTorch CUDA settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) 

    seed=69
    set_seed(seed)  # Set random seed for reproducibility

    # Display GPU information
    n_gpus = torch.cuda.device_count()
    print("IS CUDA AVAILABLE", torch.cuda.is_available())
    print("DEVICE 0 NAME", torch.cuda.get_device_name(0))
    print("TORCH VERSION", torch.__version__)
    print("NOMBRE DE GPU : ", n_gpus)

    accuracy_metric = evaluate.load("accuracy")  # Load accuracy metric

    # Load dataset using pandas
    dataset_path = 'E:/AI/projets/LLM project efrei/llm_project_M2/data/data_code.json'
    df = pd.read_json(dataset_path)

    print(f'Number of prompts: {len(df)}')
    print(f'Column names are: {df.columns.tolist()}')

    nb_samples = 3
    samples = df.sample(n=nb_samples, random_state=seed)  # Sample a few rows for display
    df_samples = pd.DataFrame(samples)
    display(df_samples)

    print(create_prompt_formats(df.iloc[0])["text"])

    # Load the pre-trained model and tokenizer
    model_path = "E:/AI/projets/LLM project efrei/llm_project_M2/model/stable-code-3b"
    model, tokenizer = load_model(model_path)
    max_length = get_max_length(model)  # Get model's max sequence length
    print("\n"+"MODEL MAX LENGTH : "+str(max_length))

    # Split the dataset into training and evaluation sets
    split_ratio = 0.15
    train_dataset = df.sample(frac=1-split_ratio, random_state=seed)  # Train set
    eval_dataset = df.drop(train_dataset.index)  # Eval set

    # Preprocessing
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)

    # Convert the processed DataFrame back to a Dataset
    train_dataset = Dataset.from_pandas(train_dataset)
    eval_dataset = Dataset.from_pandas(eval_dataset)

    # Define output directories
    output_dir = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint"
    output_merged_dir = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint_merged"

    for dir_path in [output_dir, output_merged_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Train the model with accelerate
    train_with_accelerate(model, tokenizer, train_dataset, eval_dataset, output_dir)

    # Merge weights for the final model
    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="cuda:0")
    model = model.merge_and_unload()
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)
