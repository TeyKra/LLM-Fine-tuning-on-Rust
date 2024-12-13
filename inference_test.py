import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Cache the resource-heavy model and tokenizer loading function to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path, tokenizer_path):
    """
    Loads a pre-trained causal language model and tokenizer for text generation.

    Parameters:
        model_path (str): Path to the pre-trained model directory.
        tokenizer_path (str): Path to the tokenizer directory.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # Determine the number of available GPUs
    n_gpus = torch.cuda.device_count()
    max_memory = f'{6000}MB'  # Set maximum memory usage per GPU
    # Load the pre-trained model with 4-bit quantization for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),  # Enable 4-bit quantization
        torch_dtype="auto",  # Automatically determine tensor data type
        device_map="cuda:0",  # Map model to GPU 0
        max_memory={i: max_memory for i in range(n_gpus)},  # Define max memory for each GPU
    )
    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Set the padding token to the end-of-sequence token for consistency
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Paths to the fine-tuned model and tokenizer
local_model_path = "E:/AI/projets/LLM project efrei/llm_project_M2/results/final_checkpoint_merged"
local_tokenizer_path = "E:/AI/projets/LLM project efrei/llm_project_M2/model/stable-code-3b"

# Load the fine-tuned model and tokenizer
model, tokenizer = load_model(local_model_path, local_tokenizer_path)

# Move the model to the GPU for faster inference
model.cuda()

def generate_response(prompt, model, tokenizer):
    """
    Generates a text response based on the input prompt using the loaded model.

    Parameters:
        prompt (str): The input text prompt for the model.
        model: The pre-trained causal language model.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        str: The generated text response.
    """
    # Tokenize the input prompt and move it to the model's device (GPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate text tokens using the model
    tokens = model.generate(
        **inputs,  # Pass the tokenized inputs to the model
        max_new_tokens=128,  # Limit the maximum number of new tokens
        temperature=0.2,  # Control randomness in the output
        do_sample=True,  # Enable sampling for diverse outputs
    )
    # Decode the generated tokens into a human-readable string
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

# Streamlit user interface
st.title("Stable Code 3B - Fine-tuned on Rust")  # App title
st.write("Chatbot Interface")  # App description

# Provide a text input field with a default prompt
default_text = "Calculates and prints the size of the struct Foo in bytes"
user_input = st.text_input("Enter your prompt here:", value=default_text)

# Trigger response generation when the button is clicked
if st.button("Generate Response"):
    if user_input:  # Check if the user provided input
        with st.spinner("Generating response..."):  # Show a spinner while generating response
            # Generate a response using the model
            response = generate_response(user_input, model, tokenizer)
        # Display the generated response
        st.write("Response:", response)
    else:
        # Notify the user to enter a prompt if the input field is empty
        st.write("Please enter a prompt to generate response.")
