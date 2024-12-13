# OUR PROJECT

The primary goal of this project is to refine the `stable-code-3b` model, a transformer language model with 2.7 billion parameters, specifically engineered for code comprehension and generation across diverse programming languages.

A significant challenge within this project is adapting the model to process new data efficiently while maintaining optimal performance and accuracy. The utilized dataset consists of large-scale, open-source textual and code data, which has been meticulously filtered and tailored for this task. This initiative aims to enhance code generation capabilities by methodically adjusting the model's weights using specific fine-tuning scripts. 

A particularly noteworthy challenge was to confine the fine-tuning process to the capabilities of a GeForce RTX 3080 graphics card, equipped with 10GB of VRAM, thus enabling a completely local setup without reliance on cloud-based infrastructure. Adapting such a substantial model within limited GPU memory necessitated innovative strategies and optimizations.


A series of strategic decisions were undertaken to address the model's size and ensure successful fine-tuning within constrained computational environments:

- **LoRA Technique**: The Low-Rank Adaptation (LoRA) method focused updates on select model components, reducing parameter numbers and computational demands. This facilitated efficient memory use on available hardware during fine-tuning.

- **Quantization and 4bit Precision**: The deployment of 8bit precision markedly minimized VRAM consumption, enabling the large model to function on a GeForce RTX 3080, which possesses restricted VRAM capacity.

- **Freezing Layers Technique**: By immobilizing all but the final model layers, we significantly decreased trainable parameter counts, aligning resources with the PEFT approach.

- **pytorch and cuda allocation configuration**: The configuration of this environment variable (`max_split_size_mb:128`) strategically handled memory fragmentation in CUDA allocations, mitigating potential memory limitation issues.

- **Accelerate Library**: Utilized within the current framework, it presents a huge improvement for reducing memory overhead by offloading computations and memory usage to CPU + RAM if needed
  
Faced with the primary constraint of localizing the fine-tuning process of a large model within the confines of a GeForce RTX 3080 (10GB VRAM), these techniques collectively facilitated substantial advancements. By strategically managing computational and memory demands, the project achieved comprehensive local model training, thereby broadening access to state-of-the-art AI technologies without cloud dependencies. Such advancements demonstrate the feasibility of employing advanced models on limited resource setups, opening avenues for widespread application in diverse settings.

# HOW OT USE

https://github.com/user-attachments/assets/9728e76c-1030-4c88-a82e-df3595314f0b

### Prerequisites

- Git must be installed on your computer.
- Ensure you have the necessary permissions to execute `.bat` and `.py` files.

### Inference

To perform inference with the LLM model, follow these steps:

1. **Clone the Repository**: 

   Clone the master branch of the repository using the following command:
   ```bash
   git clone https://github.com/leez7onefl/llm_project_M2.git
   ```

2. **Download Model Checkpoint**:

   Visit the [Hugging Face repository](https://huggingface.co/leez7one/stable-code-3b-fine-tuned/tree/main) to download all files in the folder `final_checkpoint_merged`. Ensure you have access to the file, and download it to your local machine. Place the complete folder directly inside the llm_project_M2 folder itself.

3. **Place the Checkpoint**:

   Move the `final_checkpoint_merged` file into the appropriate directory within the cloned repository as specified by the project documentation.

4. **Build the Project**: 

   Navigate to the repository directory and double-click on `build.bat` to set up the environment.

5. **Launch the Inference**: 

   Double-click on `launch.bat` to start inference with the model.
   
---
### Fine-tuning

To fine-tune the LLM model, follow these steps:

1. **Clone the Repository**:

   Clone the master branch of the repository using the following command:
   ```bash
   git clone https://github.com/leez7onefl/llm_project_M2.git
   ```

2. **Build the Project**:

   Navigate to the repository directory and double-click on `build.bat` to set up the environment.

3. **Run Fine-tuning Script**: 

   Execute the `fine_tune_script.py` by running:
   ```bash
   python fine_tune_script.py
   ```

# RESULTS

500 epochs
![500_steps_training_progression_plot](https://github.com/user-attachments/assets/ffc23a99-72c4-46a3-af4d-6383400a6d34)
---

1000 epochs
![1000_steps_training_progression_plot](https://github.com/user-attachments/assets/2b32861b-fc70-4ece-adba-66ca64653179)
---

1500 epochs
![1500_steps_training_progression_plot](https://github.com/user-attachments/assets/0a70716b-0862-4fd6-affc-608f5e12c6c8)
---

2000 epochs
![2000_steps_training_progression_plot](https://github.com/user-attachments/assets/176b8a5d-7d84-49be-914c-ec9fdc4b23cd)

## Additional Information

- If you encounter any issues, please check that your environment meets all prerequisites and consider reviewing error logs for more details.
- Feel free to open an issue in this repository if problems persist.
