import numpy as np
from PIL import Image
import sys
sys.path.insert(0, './goede-image-placer')
from perfect_shadow import PerfectShadow

def test_perfect_shadow():
    # Load the test image
    try:
        image = Image.open("tests/testring.png")
    except FileNotFoundError:
        print("Error: tests/testring.png not found. Make sure the test image is in the correct path.")
        return

    # Convert to numpy array
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array[np.newaxis, ...] # Add batch dimension

    # Instantiate the node
    perfect_shadow_node = PerfectShadow()

    # Test all 12 directions
    for i in range(1, 13):
        # Call the apply_shadow method
        shadow_array, = perfect_shadow_node.apply_shadow(image_array, i, 5)

        # Convert back to PIL image
        shadow_array_np = shadow_array.cpu().numpy()
        shadow_image = Image.fromarray(np.clip(255. * shadow_array_np.squeeze(), 0, 255).astype(np.uint8))

        # Save the image for manual verification
        shadow_image.save(f"test_spotlight_{i}.png")

    print("Test passed! All shadow images generated.")

if __name__ == "__main__":
    test_perfect_shadow()
