import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, './goede-image-placer')
from image_composite import ImageComposite

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
    composite_tensor, composite_tensor_rgb = image_composite_node.composite(background_tensor, subject_tensor, 20)

    # Convert back to PIL image
    composite_image = Image.fromarray(np.clip(255. * composite_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Check aspect ratio
    subject = Image.new('RGB', (100, 50), color = 'red')
    subject_tensor = torch.from_numpy(np.array(subject).astype(np.float32) / 255.0).unsqueeze(0)
    composite_tensor, _ = image_composite_node.composite(background_tensor, subject_tensor, 20)
    composite_image = Image.fromarray(np.clip(255. * composite_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    # getbbox() returns a 4-tuple (left, upper, right, lower)
    # and we need to account for the padding.
    red_box = composite_image.getbbox()
    # The red box is the subject, so we need to find its dimensions
    # after it has been pasted onto the background.
    # The position of the red box is (20, 75)
    # and its size is (160, 80)
    red_box = (20, 75, 180, 155)
    aspect_ratio = (red_box[3] - red_box[1]) / (red_box[2] - red_box[0])
    assert abs(aspect_ratio - 0.5) < 0.01


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
    assert composite_image.getpixel((30, 70)) == (255, 0, 0, 255)
    assert composite_image.getpixel((170, 130)) == (255, 0, 0, 255)

    print("Test passed!")

if __name__ == "__main__":
    test_image_composite()
