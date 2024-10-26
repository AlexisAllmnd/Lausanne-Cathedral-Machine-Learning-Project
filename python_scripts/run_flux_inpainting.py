import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
from peft import PeftModel
from PIL import PngImagePlugin
import os
from datetime import datetime

# Set the model, LoRA, and device configuration
model_id = "black-forest-labs/FLUX.1-schnell"
device = "cuda" if torch.cuda.is_available() else "cpu"

guidance_scale = 5  # Adjust guidance scale for better prompt consistency
num_inference_steps = 25  # Number of diffusion steps
strength = 0.8  # Strength of inpainting for blending

batch_size = 3  # Images per batch
num_batches = 4  # Number of batches to generate

# Seed settings
generator = torch.Generator(device)

# LoRA parameters (optional)
use_lora = False  # Enable if you wish to use LoRA
lora_params = {
    "alpha": 0.75,
    "r": 4,
    "beta": 0.01,
    "dropout": 0.1
}

# Load the Flux inpainting pipeline
pipe = FluxInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to(device)

# Load the LoRA model if enabled
if use_lora:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, model_id, alpha=lora_params["alpha"], r=lora_params["r"], beta=lora_params["beta"], dropout=lora_params["dropout"])
    pipe.unet.to(device)

# Create a directory with the current date and time for saving images
output_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(output_dir, exist_ok=True)

# Load source image and mask
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

source_image = load_image(img_url)
mask_image = load_image(mask_url)

# Define inpainting prompt and negative prompt
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
negative_prompt = "low resolution, blurry, distorted face"

# Function to save images with metadata
def save_image_with_metadata(image, image_path, metadata):
    png_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    image.save(image_path, "PNG", pnginfo=png_info)

# Batch generation loop
for batch in range(num_batches):
    print(f"Generating batch {batch + 1}/{num_batches}...")
    batch_seed = generator.seed()  # Get a unique seed per batch

    for i in range(batch_size):
        generator.manual_seed(batch_seed)

        # Run the inpainting with prompt and negative prompt
        image = pipe(
            prompt=prompt,
            image=source_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator
        ).images[0]

        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": batch_seed,
            "model_id": model_id
        }

        # Save the image with metadata
        image_path = os.path.join(output_dir, f"flux_inpainting_result_batch{batch+1}_image{i+1}_seed{batch_seed}.png")
        save_image_with_metadata(image, image_path, metadata)
        print(f"Image saved: {image_path}")
        batch_seed += 1

print("Batch generation completed.")
