from PIL import Image

# Load the image and check its metadata
img = Image.open("/Users/alexisallemand/Documents/GitHub/Lausanne-Cathedral-Machine-Learning-Project/Results/PIC1/Step3/FromCrowned1/FromCrowned1-6.png")
metadata = img.info
print(metadata)