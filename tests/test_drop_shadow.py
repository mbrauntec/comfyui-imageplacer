import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, './goede-image-placer')
from drop_shadow import DropShadow

def test_drop_shadow():
    # Create a dummy image
    image = Image.new('RGBA', (200, 200), color = (255, 0, 0, 255))

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    # Instantiate the node
    drop_shadow_node = DropShadow()

    # Call the add_shadow method
    shadow_tensor, = drop_shadow_node.add_shadow(image_tensor, 90, 50, 20, 1, "#000000")

    # Convert back to PIL image
    shadow_image = Image.fromarray(np.clip(255. * shadow_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Save the image for manual verification
    shadow_image.save("test_drop_shadow.png")

    # Assertions
    assert shadow_image.width == 200
    assert shadow_image.height == 200

    print("Test passed!")

if __name__ == "__main__":
    test_drop_shadow()
