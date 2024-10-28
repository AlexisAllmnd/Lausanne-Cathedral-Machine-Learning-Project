import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
from PIL import PngImagePlugin
import os
from datetime import datetime

# Set the model, LoRA, and device configuration
model_id = "black-forest-labs/FLUX.1-schnell"  # FLUX model
device = "cuda" if torch.cuda.is_available() else "cpu"

guidance_scale = 2  # Adjust guidance scale for better prompt consistency
num_inference_steps = 30  # Number of diffusion steps
strength = 0.6  # Strength of inpainting for blending

batch_size = 2  # Images per batch
num_batches = 8  # Number of batches to generate

# Seed settings
generator = torch.Generator(device)

# Load the Flux inpainting pipeline
pipe = FluxInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to(device)

# Create a directory with the current date and time for saving images
output_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(output_dir, exist_ok=True)

# Load source image and mask
img_url = "https://github.com/AlexisAllmnd/Lausanne-Cathedral-Machine-Learning-Project/blob/main/Results/PIC1/Step%202/CrownedPreProcessed1.2.png?raw=true"
mask_url = "https://github.com/AlexisAllmnd/Lausanne-Cathedral-Machine-Learning-Project/blob/main/Results/PIC1/Step3/Masks/maskCrowned2.png?raw=true"

source_image = load_image(img_url)
mask_image = load_image(mask_url)

blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor=20)


# Define improved inpainting prompt with stone crown detail
prompt = (
    "Generate a serene, 13th-century Gothic Virgin Mary statue head in weathered limestone. Features include closed lips, soft gaze, and smooth lines. Add a simple headdress, subtle wear, and light texturing. Emphasize gentle contours, with balanced lighting to capture a peaceful expression and aged stone detail."
)

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

        # Run the inpainting with the prompt
        image = pipe(
            prompt=prompt,
            image=source_image,
            mask_image=mask_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
