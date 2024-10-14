import torch
from diffusers import FluxInpaintPipeline
from diffusers.utils import load_image
from peft import PeftModel
from PIL import PngImagePlugin

# Set the model, LoRA, and device configuration
model_id = "black-forest-labs/FLUX.1-dev"  # Use your chosen FLUX model
device = "cuda" if torch.cuda.is_available() else "cpu"

guidance_scale = 5
num_inference_steps = 25

batch_size = 3  # Number of images to generate per batch
num_batches = 4  # Number of batches
strength = 0.8  # Strength of inpainting for better blending

# Seed settings
generator = torch.Generator(device)

# Set the LoRA parameters
use_lora = False  # Set to True if you want to use LoRA
lora_params = {
    "alpha": 0.75,  # Strength of the LoRA model
    "r": 4,  # Rank used in the LoRA decomposition
    "beta": 0.01,  # Regularization parameter for LoRA
    "dropout": 0.1  # Dropout rate for LoRA layers
}

# Load the Flux inpainting pipeline without custom schedulers
pipe = FluxInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to(device)

# Optionally, load the LoRA model for fine-tuning
if use_lora:
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model_id, alpha=lora_params["alpha"], r=lora_params["r"], beta=lora_params["beta"], dropout=lora_params["dropout"])
    pipe.unet.to(device)

# Load the images (source image and mask) using load_image
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

source_image = load_image(img_url)
mask_image = load_image(mask_url)

# Define your inpainting prompt and negative prompt
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
negative_prompt = "low resolution, blurry, distorted face"

# Function to save metadata in the image
def save_image_with_metadata(image, image_path, metadata):
    # Save image with embedded metadata in PNG
    png_info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, str(value))

    image.save(image_path, "PNG", pnginfo=png_info)

# Batch image generation loop
for batch in range(num_batches):

    batch_seed = generator.seed()  # Generates a random seed from the generator
    # Uncomment the next line to fix the seed manually:
    # batch_seed = generator.manual_seed(seed)

    print(f"Generating batch {batch + 1}/{num_batches}...")
    for i in range(batch_size):
        # Run the inpainting process with negative prompt support and generator for seed
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
        image_path = f"flux_inpainting_result_batch{batch+1}_image{i+1}_seed{batch_seed}.png"
        save_image_with_metadata(image, image_path, metadata)
        print(f"Image saved: {image_path}")
        batch_seed+=1

print("Batch generation completed.")
