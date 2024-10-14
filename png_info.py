from PIL import Image

# Load the image and check its metadata
img = Image.open("/content/2024-10-14_15-11-32/sdxl_inpainting_result_batch2_image1_seed2851062217407373.png")
metadata = img.info
print(metadata)