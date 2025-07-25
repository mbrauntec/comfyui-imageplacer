from PIL import Image, ImageFilter
import numpy as np
import math
import torch

class PerfectShadow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "light_from": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 12,
                    "step": 1,
                    "display": "slider"
                }),
                "shadow_length": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider"
                }),
                "shrink": ("INT", {
                    "default": 0,
                    "min": -5,
                    "max": 5,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_shadow"

    CATEGORY = "Goede"

    def apply_shadow(self, image, light_from, shadow_length, shrink):
        # The input is a tensor, but we will treat it as a numpy array
        # and convert it to a PIL image.
        if hasattr(image, 'cpu'):
            image = image.cpu().numpy()
        image_np = np.clip(255. * image.squeeze(), 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        # Ensure image is RGBA
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        # Create a silhouette
        alpha = image_pil.getchannel('A')
        shadow_color = (0, 0, 0, 255)
        shadow = Image.new('RGBA', image_pil.size, shadow_color)
        shadow.putalpha(alpha)

        # Shrink the shadow mask
        if shrink != 0:
            width, height = shadow.size
            if shrink < 0:
                new_width = int(width * (1 + shrink / 10.0))
                shadow = shadow.resize((new_width, height), Image.LANCZOS)
            else:
                new_height = int(height * (1 - shrink / 10.0))
                shadow = shadow.resize((width, new_height), Image.LANCZOS)

        # Shadow parameters
        shadow_length = shadow_length * 100  # A large value to create a long shadow
        blur_radius = 10

        # Angle mapping from clock hour to degrees
        angle_map = {
            1: 150, 2: 120, 3: 90, 4: 60, 5: 30, 6: 0,
            7: 330, 8: 300, 9: 270, 10: 240, 11: 210, 12: 180
        }
        angle = angle_map[light_from]
        angle_rad = math.radians(angle)

        # The direction of the shadow is opposite to the light source
        shadow_angle_rad = angle_rad + math.pi

        # Create a long shadow by shearing the image
        x_shear = math.cos(shadow_angle_rad)
        y_shear = math.sin(shadow_angle_rad)

        # We create a new image large enough to hold the sheared shadow
        new_width = image_pil.width + abs(int(shadow_length * x_shear))
        new_height = image_pil.height + abs(int(shadow_length * y_shear))

        long_shadow = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))

        # Paste the silhouette multiple times to create the long shadow effect
        for i in range(shadow_length):
            x_offset = int(i * x_shear)
            y_offset = int(i * y_shear)

            # Adjust position to keep the shadow within the new canvas
            paste_x = (new_width - image_pil.width) // 2 + x_offset
            paste_y = (new_height - image_pil.height) // 2 + y_offset

            long_shadow.paste(shadow, (paste_x, paste_y), shadow)

        # Blur the shadow
        if blur_radius > 0:
            long_shadow = long_shadow.filter(ImageFilter.GaussianBlur(blur_radius))

        # Composite the original image over the shadow
        # The original image should be centered in the new canvas
        img_x = (new_width - image_pil.width) // 2
        img_y = (new_height - image_pil.height) // 2

        final_image = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 0))
        final_image.paste(long_shadow, (0,0), long_shadow)
        final_image.paste(image_pil, (img_x, img_y), image_pil)

        # Convert back to numpy array, which is what the test expects
        final_array = np.array(final_image).astype(np.float32) / 255.0
        # We need to add the batch dimension back
        return (torch.from_numpy(final_array[np.newaxis, ...]),)

NODE_CLASS_MAPPINGS = {
    "PerfectShadow": PerfectShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerfectShadow": "Perfect Shadow"
}
