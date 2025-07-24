import torch
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import math

class Spotlight:
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
                "shadow_length": ("FLOAT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 0.1,
                    "display": "slider"
                }),
                 "shadow_blur": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "slider"
                }),
                "shadow_color": ("STRING", {
                    "default": "#000000"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_spotlight"

    CATEGORY = "Goede"

    def apply_spotlight(self, image, light_from, shadow_length, shadow_blur, shadow_color):
        # Convert tensor to PIL image
        image_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        # Ensure image is RGBA
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')

        # Extract alpha channel to create shadow
        alpha = image_pil.getchannel('A')
        shadow = Image.new('RGBA', image_pil.size, color=shadow_color)
        shadow.putalpha(alpha)

        shadow_scale = shadow_length / 5.0

        # Schatten ggf. skalieren (um Mittelpunkt)
        if shadow_scale != 1.0:
            cx, cy = image_pil.width // 2, image_pil.height // 2
            new_w = int(shadow.width * shadow_scale)
            new_h = int(shadow.height * shadow_scale)
            shadow = shadow.resize((new_w, new_h), Image.LANCZOS)
            scale_offset_x = cx - new_w // 2
            scale_offset_y = cy - new_h // 2
        else:
            scale_offset_x = 0
            scale_offset_y = 0

        # Schatten weichzeichnen
        if shadow_blur > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

        # Convert light_from (1-12) to an angle in degrees
        light_angle_map = {
            1: 30, 2: 60, 3: 90, 4: 120, 5: 150, 6: 180,
            7: 210, 8: 240, 9: 270, 10: 300, 11: 330, 12: 360
        }
        light_angle = light_angle_map[light_from]

        # The shadow is cast in the opposite direction of the light
        shadow_dir = (light_angle + 180) % 360

        # Offset in Schattenrichtung berechnen
        angle_rad_shadow = math.radians(shadow_dir)
        dx_shadow = math.cos(angle_rad_shadow)
        dy_shadow = math.sin(angle_rad_shadow)

        # The shadow distance is controlled by the shadow_length
        shadow_distance = (shadow_length - 5) * 20

        offset_x = int(round(dx_shadow * shadow_distance))
        offset_y = int(round(dy_shadow * shadow_distance))

        # Gesamt-Offset: Skalierung + Richtung
        total_offset_x = scale_offset_x + offset_x
        total_offset_y = scale_offset_y + offset_y

        # Neue Bildgröße berechnen, damit alles reinpasst
        min_x = min(0, total_offset_x)
        min_y = min(0, total_offset_y)
        max_x = max(image_pil.width, total_offset_x + shadow.width)
        max_y = max(image_pil.height, total_offset_y + shadow.height)
        composite_width = max_x - min_x
        composite_height = max_y - min_y

        composite_image = Image.new("RGBA", (composite_width, composite_height), (0, 0, 0, 0))

        # Schatten einfügen
        shadow_x = total_offset_x - min_x
        shadow_y = total_offset_y - min_y
        composite_image.paste(shadow, (shadow_x, shadow_y), shadow)

        # Originalbild einfügen (immer mittig)
        image_x = -min_x
        image_y = -min_y
        composite_image.paste(image_pil, (image_x, image_y), image_pil)

        # Convert the composite image back to a tensor
        composite_tensor = torch.from_numpy(np.array(composite_image).astype(np.float32) / 255.0).unsqueeze(0)
        return (composite_tensor,)

NODE_CLASS_MAPPINGS = {
    "Spotlight": Spotlight
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Spotlight": "Spotlight"
}
