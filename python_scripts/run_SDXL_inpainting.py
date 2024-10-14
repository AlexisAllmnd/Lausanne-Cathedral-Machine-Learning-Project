import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from PIL import PngImagePlugin
import os
from datetime import datetime

# Set the model and device configuration
model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL model
device = "cuda" if torch.cuda.is_available() else "cpu"

guidance_scale = 5  # Adjust guidance for better consistency with the prompt
num_inference_steps = 30  # More steps for a refined image
strength = 0.8  # Strength of inpainting for better blending

batch_size = 2  # Number of images to generate per batch
num_batches = 5  # Number of batches

# Seed settings
generator = torch.Generator(device)

# Load the SDXL Turbo Inpainting pipeline (default scheduler)
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to(device)

# Create a directory with the current date and time
output_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(output_dir, exist_ok=True)

# Load the images (source image and mask)
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

source_image = load_image(img_url)
mask_image = load_image(mask_url)

# Define the inpainting prompt and negative prompt
prompt = "A yellow cat, sitting on a park bench, realistic proportions, natural blending with the bench, high quality, photorealistic"
negative_prompt = "low resolution, blurry, distorted face"

# Function to save metadata in the image
def save_image_with_metadata(image, image_path, metadata):
    png_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))
    image.save(image_path, "PNG", pnginfo=png_info)

# Batch image generation loop
for batch in range(num_batches):
    print(f"Generating batch {batch + 1}/{num_batches}...")
    batch_seed = generator.seed()  # Generate a unique seed for each batch

    for i in range(batch_size):
        # Fix seed for each image in the batch for consistency
        generator.manual_seed(batch_seed)

        # Run the inpainting process
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
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
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": batch_seed,
            "model_id": model_id
        }

        # Save the result image with metadata
        image_path = os.path.join(output_dir, f"sdxl_inpainting_result_batch{batch+1}_image{i+1}_seed{batch_seed}.png")
        save_image_with_metadata(image, image_path, metadata)
        print(f"Image saved: {image_path}")
        batch_seed+=1

print("Batch generation completed.")
