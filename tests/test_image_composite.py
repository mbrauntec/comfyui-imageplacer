import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')
from custom_nodes.image_composite import ImageComposite

def test_image_composite():
    # Create dummy images
    background = Image.new('RGB', (200, 200), color = 'blue')
    subject = Image.new('RGB', (100, 100), color = 'red')

    # Convert to tensors
    background_tensor = torch.from_numpy(np.array(background).astype(np.float32) / 255.0).unsqueeze(0)
    subject_tensor = torch.from_numpy(np.array(subject).astype(np.float32) / 255.0).unsqueeze(0)

    # Instantiate the node
    image_composite_node = ImageComposite()

    # Call the composite method
    composite_tensor, = image_composite_node.composite(background_tensor, subject_tensor, 20)

    # Convert back to PIL image
    composite_image = Image.fromarray(np.clip(255. * composite_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Save the image for manual verification
    composite_image.save("test_composite.png")

    # Assertions
    assert composite_image.width == 200
    assert composite_image.height == 200

    # Check colors
    # a few pixels from the border
    assert composite_image.getpixel((10, 10)) == (0, 0, 255, 255)
    assert composite_image.getpixel((190, 190)) == (0, 0, 255, 255)
    # a few pixels from the subject
    assert composite_image.getpixel((30, 30)) == (255, 0, 0, 255)
    assert composite_image.getpixel((170, 170)) == (255, 0, 0, 255)

    print("Test passed!")

if __name__ == "__main__":
    test_image_composite()
