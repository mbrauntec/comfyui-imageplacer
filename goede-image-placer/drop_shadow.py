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
                    "default": 6,
                    "min": 0,
                    "max": 12,
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

        # --- Konturpunkt an der gewünschten Uhrzeit-Position finden ---
        # shadow_angle als Uhrzeit (1–12), 12 Uhr = oben, 3 Uhr = rechts, 6 Uhr = unten, 9 Uhr = links
        # Umrechnung: 12 Uhr = 270°, 3 Uhr = 0°, 6 Uhr = 90°, 9 Uhr = 180°
        light_angle = (shadow_angle - 3) * 30
        # Schattenrichtung = Lichtquelle + 180°
        shadow_dir = (light_angle + 180) % 360
        # Startpunkt auf der Kontur in Lichtquellenrichtung suchen (also light_angle, nicht shadow_dir!)
        angle_rad = math.radians(light_angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)
        alpha_np = np.array(alpha)
        h, w = alpha_np.shape
        cx, cy = w // 2, h // 2
        # Vom Mittelpunkt nach außen in Lichtquellenrichtung laufen, bis Alpha=0
        max_radius = int(1.5 * max(cx, cy))
        edge_x, edge_y = cx, cy
        for r in range(0, max_radius):
            x = int(round(cx + r * dx))
            y = int(round(cy + r * dy))
            if 0 <= x < w and 0 <= y < h:
                if alpha_np[y, x] == 0:
                    # Letzter Punkt mit Alpha>0 ist die Kontur
                    edge_x = int(round(cx + (r-1) * dx))
                    edge_y = int(round(cy + (r-1) * dy))
                    break
        # --- Schatten perspektivisch verzerren (elliptisch) ---
        ellipse_scale = 0.4  # ggf. anpassen
        shadow = shadow.transform(
            (shadow.width, int(shadow.height * ellipse_scale)),
            Image.AFFINE,
            (1, 0, 0, 0, ellipse_scale, 0),
            resample=Image.BICUBIC
        )
        # Nach Stauchung Offset anpassen
        scale_offset_y = int(scale_offset_y * ellipse_scale)
        # Nach Skalierung und Stauchung Kantenpunkt anpassen
        if shadow_scale != 1.0:
            new_w = shadow.width
            new_h = shadow.height
            new_cx, new_cy = new_w // 2, int(new_h // 2)
            edge_x = int((edge_x - cx) * shadow_scale + new_cx)
            edge_y = int((edge_y - cy) * shadow_scale * ellipse_scale + new_cy)
        else:
            edge_y = int(edge_y * ellipse_scale)
        # Offset in Schattenrichtung berechnen (wie gehabt)
        angle_rad_shadow = math.radians(shadow_dir)
        dx_shadow = math.cos(angle_rad_shadow)
        dy_shadow = math.sin(angle_rad_shadow)
        offset_x = int(round(dx_shadow * shadow_distance))
        offset_y = int(round(dy_shadow * shadow_distance))
        # Gesamt-Offset: Kantenpunkt + Richtung + Skalierung
        total_offset_x = scale_offset_x + (edge_x - image_pil.width // 2) + offset_x
        total_offset_y = scale_offset_y + (edge_y - image_pil.height // 2) + offset_y
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
    "DropShadow": DropShadow
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DropShadow": "Drop Shadow"
}
