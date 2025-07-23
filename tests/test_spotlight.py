import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, './goede-image-placer')
from spotlight import Spotlight

def test_spotlight():
    # Create a dummy image
    image = Image.new('RGBA', (100, 100), (255, 0, 0, 255))

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    # Instantiate the node
    spotlight_node = Spotlight()

    # Call the apply_spotlight method
    spotlight_tensor, = spotlight_node.apply_spotlight(image_tensor, 12, 10, 20, "#000000")

    # Convert back to PIL image
    spotlight_image = Image.fromarray(np.clip(255. * spotlight_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Save the image for manual verification
    spotlight_image.save("test_spotlight.png")

    # Assertions
    assert spotlight_image.width == 200
    assert spotlight_image.height > 100

    print("Test passed!")

if __name__ == "__main__":
    test_spotlight()
