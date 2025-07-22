import torch
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import math

class DropShadow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_angle": ("INT", {
                    "default": 135,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                    "display": "slider"
                }),
                "shadow_distance": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "slider"
                }),
                "shadow_blur": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "slider"
                }),
                "shadow_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "shadow_color": ("STRING", {
                    "default": "#000000"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_shadow"

    CATEGORY = "Image"

    def add_shadow(self, image, shadow_angle, shadow_distance, shadow_blur, shadow_scale, shadow_color):
        # Convert tensor to PIL image
        image_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        # Ensure image is RGBA
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        # Extract alpha channel to create shadow
        alpha = image_pil.getchannel('A')
        shadow = Image.new('RGBA', image_pil.size, color=shadow_color)
        shadow.putalpha(alpha)

        # Scale the shadow
        if shadow_scale != 1.0:
            shadow = shadow.resize((int(shadow.width * shadow_scale), int(shadow.height * shadow_scale)), Image.LANCZOS)

        # Blur the shadow
        if shadow_blur > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

        # Calculate shadow offset
        angle_rad = math.radians(shadow_angle)
        offset_x = int(shadow_distance * math.cos(angle_rad))
        offset_y = int(shadow_distance * math.sin(angle_rad))

        # Adjust offset for scaled shadow
        offset_x -= (shadow.width - image_pil.width) // 2
        offset_y -= (shadow.height - image_pil.height) // 2

        # Create a new image for the final composite
        composite_image = Image.new("RGBA", image_pil.size)

        # Paste shadow
        composite_image.paste(shadow, (offset_x, offset_y), shadow)

        # Paste original image on top
        composite_image.paste(image_pil, (0, 0), image_pil)

        # Convert the composite image back to a tensor
        composite_tensor = torch.from_numpy(np.array(composite_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (composite_tensor,)

NODE_CLASS_MAPPINGS = {
    "DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DropShadow": "Drop Shadow"
}
