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

    CATEGORY = "Goede"

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

        # --- Kante des Objekts in Schattenrichtung finden ---
        # Alpha-Kanal als numpy-Array
        alpha_np = np.array(alpha)
        h, w = alpha_np.shape
        angle_rad = math.radians(shadow_angle)
        dx = math.cos(angle_rad)
        dy = -math.sin(angle_rad)
        # Für alle Pixel mit Alpha > 0: Skalarprodukt mit Richtungsvektor berechnen
        ys, xs = np.nonzero(alpha_np > 0)
        if len(xs) > 0:
            # Skalarprodukt = Projektion auf Richtungsvektor
            projections = xs * dx + ys * dy
            idx = np.argmax(projections)
            edge_x = xs[idx]
            edge_y = ys[idx]
        else:
            # Fallback: Mittelpunkt
            edge_x = w // 2
            edge_y = h // 2

        # --- Schatten um Mittelpunkt skalieren ---
        if shadow_scale != 1.0:
            cx, cy = image_pil.width // 2, image_pil.height // 2
            new_w = int(shadow.width * shadow_scale)
            new_h = int(shadow.height * shadow_scale)
            shadow = shadow.resize((new_w, new_h), Image.LANCZOS)
            new_cx, new_cy = new_w // 2, new_h // 2
            scale_offset_x = cx - new_cx
            scale_offset_y = cy - new_cy
            # Skalierten Kantenpunkt berechnen
            edge_x = int((edge_x - cx) * shadow_scale + new_cx)
            edge_y = int((edge_y - cy) * shadow_scale + new_cy)
        else:
            scale_offset_x = 0
            scale_offset_y = 0

        # --- Spiegelung je nach Winkel ---
        if 90 < shadow_angle < 270:
            shadow = ImageOps.mirror(shadow)
            edge_x = shadow.width - 1 - edge_x
        if 180 < shadow_angle < 360:
            shadow = ImageOps.flip(shadow)
            edge_y = shadow.height - 1 - edge_y

        # Blur the shadow
        if shadow_blur > 0:
            shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

        # Schatten-Offset: Abstand in Richtung + Kantenpunkt
        offset_x = int(shadow_distance * dx)
        offset_y = int(shadow_distance * dy)
        # Gesamtoffset: Kantenpunkt + Richtung + Skalierung
        total_offset_x = offset_x + scale_offset_x + (edge_x - image_pil.width // 2)
        total_offset_y = offset_y + scale_offset_y + (edge_y - image_pil.height // 2)

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

        # Originalbild einfügen
        image_x = -min_x
        image_y = -min_y
        composite_image.paste(image_pil, (image_x, image_y), image_pil)

        # Convert the composite image back to a tensor
        composite_tensor = torch.from_numpy(np.array(composite_image).astype(np.float32) / 255.0).unsqueeze(0)

        return (composite_tensor,)

NODE_CLASS_MAPPINGS = {
    "DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DropShadow": "Drop Shadow"
}
