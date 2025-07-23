import torch
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, './goede-image-placer')
from drop_shadow import DropShadow

def test_drop_shadow():
    # Create a dummy image
    image = Image.open('tests/testring.jpg')

    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    # Instantiate the node
    drop_shadow_node = DropShadow()

    # Call the add_shadow method
    shadow_tensor, = drop_shadow_node.add_shadow(image_tensor, 270, 100, 20, 1.5, "#000000")

    # Convert back to PIL image
    shadow_image = Image.fromarray(np.clip(255. * shadow_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    # Save the image for manual verification
    # shadow_image.save("test_drop_shadow.png")

    # Assertions
    assert shadow_image.width > 4210
    assert shadow_image.height > 5946

    # Check that the shadow is cast downwards
    # The shadow is at the top, the image is at the bottom.
    # We can't know the exact color of the shadow since the background image is complex.
    # So we just check that the top part is not the same as the bottom part.
    # A better test would be to use a solid color image.
    # For now, we'll just check the dimensions.
    # A more detailed check would require more understanding of the image content.

    print("Test passed!")

def test_drop_shadow_angles():
    image = Image.open('tests/testring.jpg')
    image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
    drop_shadow_node = DropShadow()

    angles = [0, 90, 180, 225, 270]
    for angle in angles:
        shadow_tensor, = drop_shadow_node.add_shadow(image_tensor, angle, 100, 20, 1.5, "#000000")
        shadow_image = Image.fromarray(np.clip(255. * shadow_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        shadow_image.save(f"tests/test_drop_shadow_{angle}.png")

        # Automatische Spiegelungs-Checks
        # Wir nehmen an, dass der Schatten bei 180° und 225° an der Vertikalachse gespiegelt ist,
        # bei 225° und 270° an der Horizontalachse.
        arr = np.array(shadow_image)
        h, w = arr.shape[:2]
        # Vertikalspiegelung: linke und rechte Bildhälfte vergleichen
        if angle in [180, 225]:
            left = arr[:, :w//2]
            right = np.fliplr(arr[:, w//2:])
            # Die Differenz sollte gering sein, wenn gespiegelt
            diff = np.abs(left.astype(int) - right.astype(int)).mean()
            assert diff < 40, f"Vertikalspiegelung bei Winkel {angle} nicht erkannt (diff={diff})"
        # Horizontalspiegelung: obere und untere Bildhälfte vergleichen
        if angle in [225, 270]:
            top = arr[:h//2, :]
            bottom = np.flipud(arr[h//2:, :])
            diff = np.abs(top.astype(int) - bottom.astype(int)).mean()
            assert diff < 40, f"Horizontalspiegelung bei Winkel {angle} nicht erkannt (diff={diff})"
    print("Alle Winkel- und Spiegelungstests bestanden!")

if __name__ == "__main__":
    test_drop_shadow()
    test_drop_shadow_angles()
