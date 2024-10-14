# Lausanne Cathedral Machine Learning Reconstruction Project

## Project Overview

The goal of this project is to reconstruct the **Virgin and Child statue** from Lausanne Cathedral using advanced **diffusion models**. This repository contains the necessary scripts, data, and configurations required for the project. By leveraging the computational power of **SCITAS** (Scientific IT & Application Support) and the supercomputer **Kuma**, we aim to create high-quality inpainted models of the missing parts of the statue.

The primary models used in this project are part of the Hugging Face `diffusers` library:
- **Flux Model:** [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Stable Diffusion XL:** [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

## Project Structure

- **`python_script/`**: This folder contains all Python scripts used to run various diffusion models for inpainting and generating outputs based on the Lausanne Cathedral statue.
  
- **`slurm_scripts/`**: The Slurm scripts necessary to run the Python scripts on the SCITAS cluster are stored here. These scripts handle job scheduling and ensure that the models run efficiently on the supercomputer.

**`data/`**: This directory contains the images used for training the LoRA model and the reference images required for the inpainting process.

- **`png_info.png`**: This file is used to store and retrieve the **metadata** associated with each generated image. The metadata includes all the parameters used during the image generation process, allowing us to track and reproduce the results if needed.

## Inpainting Process and Diffusion Models

Inpainting is a technique where parts of an image that are missing or damaged are filled in by AI models. In our project, we are leveraging diffusion models, which work by gradually "denoising" an image from random noise into a plausible version based on a prompt or guidance.

To perform **inpainting** using Hugging Face’s `diffusers` library, we rely on models that take an incomplete image (with a mask over the missing parts) and a prompt (describing what the missing content should be) to generate the missing parts. You can read more about the inpainting process in detail [here](https://huggingface.co/docs/diffusers/en/using-diffusers/inpaint).

Below is a typical setup for inpainting using Stable Diffusion from Hugging Face:

```python
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import requests

# Load the model
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe = pipe.to("cuda")

# Load the source image and mask
image = Image.open("path_to_incomplete_statue_image.png").convert("RGB")
mask = Image.open("path_to_mask.png").convert("L")

# Inpainting prompt
prompt = "A classical statue of the Virgin Mary holding a child, in Gothic style, delicate features."

# Run the inpainting process
result = pipe(prompt=prompt, image=image, mask_image=mask, guidance_scale=7.5).images[0]
result.save("reconstructed_statue.png")
```

## Inpainting for the Lausanne Cathedral Statue

In this project, inpainting techniques will be applied to reconstruct the missing head and child from the original **Virgin and Child statue** of Lausanne Cathedral. We are utilizing **high-resolution images** from the Lausanne Cathedral dataset to feed into advanced diffusion models, which will complete the missing parts of the statue.

## SCITAS and Kuma Supercomputer

The **SCITAS** infrastructure at EPFL provides access to high-performance computing resources. Our project leverages the power of the **Kuma** supercomputer, designed for demanding tasks like machine learning and AI. SCITAS allows us to run complex diffusion models in parallel, greatly accelerating the inpainting and model training processes.

In the `slurm_scripts/` directory, you will find the Slurm job scripts that schedule and manage these large-scale computational tasks on SCITAS. These jobs efficiently distribute the workload across multiple GPUs, enabling faster and more efficient training and inference.

## LoRA (Low-Rank Adaptation) and Model Training

The next phase of the project involves creating a **LoRA** (Low-Rank Adaptation) model based on the collected dataset. LoRA is a method that allows us to fine-tune large models with fewer resources. Instead of retraining the entire model from scratch, LoRA enables us to adapt a pre-trained model for specific tasks, such as generating highly detailed Gothic statues, by only adjusting a small subset of parameters.

### What is LoRA?

- LoRA injects **low-rank matrices** into the layers of a pre-trained model.
- These matrices modify a small subset of the model’s weights, while preserving the majority of the model’s pre-trained knowledge.
- The result is **faster training**, fewer required resources, and improved adaptability to specific details, like the intricate features of the Lausanne Cathedral statues.

In this project, we will use LoRA to create a custom model that can generate accurate and refined representations of the **Gothic architecture and artistry** of the Virgin and Child statue. Once created, this LoRA model can be used in combination with any diffusion model for more precise inpainting results.

## How to Run the Scripts

To run the models on SCITAS, follow these steps:

1. **Clone the Repository:**
```bash
git clone https://github.com/your_username/Lausanne-Cathedral-Project.git
cd Lausanne-Cathedral-Project
```

2. **Submit a Job**
Navigate to the `slurm_scripts/` directory and submit a job to the SCITAS cluster:
```bash
sbatch run_model.slurm
```

3. **Check Results:** After the job is complete, the results will be saved in the specified output directory, and the reconstructed images will be available for review.


