from PIL import Image
import numpy as np

# Load the t-shirt image (make sure the image is named 'tshirt.png' and is in the same directory)
input_image_path = 'T-Shirt-PNG-Photo.png'
output_image_path = 'tshirt_colored.png'

# Sample RGB color (blue)
target_color = (255, 255, 0,0.8)

# Open the image
image = Image.open(input_image_path).convert('RGBA')

# Convert image to numpy array
img_array = np.array(image)

# Create a mask for the t-shirt (non-transparent and non-black areas)
# You may need to adjust the mask depending on your image background
mask = (img_array[..., 0] > 20) | (img_array[..., 1] > 20) | (img_array[..., 2] > 20)

# Change color where mask is True
img_array[mask, 0] = target_color[0]
img_array[mask, 1] = target_color[1]
img_array[mask, 2] = target_color[2]

# Convert back to image
colored_image = Image.fromarray(img_array, 'RGBA')

# Save the result
colored_image.save(output_image_path)

print(f'T-shirt color changed and saved as {output_image_path}')
