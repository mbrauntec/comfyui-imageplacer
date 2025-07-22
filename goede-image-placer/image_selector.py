import os
import torch
import numpy as np
from PIL import Image

class ImageSelector:
    @classmethod
    def INPUT_TYPES(s):
        image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        return {
            "required": {
                "image": (image_files, ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select_image"

    CATEGORY = "Goede"

    def select_image(self, image):
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images", image)
        i = Image.open(image_path)
        i = i.convert("RGB")
        image = np.array(i).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

NODE_CLASS_MAPPINGS = {
    "ImageSelector": ImageSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSelector": "Goede Image Selector"
}
