import torch
from PIL import Image
import numpy as np

class AddPadding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "slider"
                }),
                "top": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "slider"
                }),
                "right": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "slider"
                }),
                "bottom": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image_4_channel", "image_3_channel",)
    FUNCTION = "add_padding"

    CATEGORY = "Image"

    def add_padding(self, image, left, top, right, bottom):
        # Convert tensor to PIL image
        pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        # Calculate new dimensions
        new_width = pil_image.width + left + right
        new_height = pil_image.height + top + bottom

        # Create a new image with transparent background
        padded_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Paste the original image
        padded_image.paste(pil_image, (left, top))

        # Convert to 4-channel tensor
        image_4_channel_tensor = torch.from_numpy(np.array(padded_image).astype(np.float32) / 255.0).unsqueeze(0)

        # Convert to 3-channel image
        image_3_channel = padded_image.convert("RGB")
        image_3_channel_tensor = torch.from_numpy(np.array(image_3_channel).astype(np.float32) / 255.0).unsqueeze(0)

        return (image_4_channel_tensor, image_3_channel_tensor,)

NODE_CLASS_MAPPINGS = {
    "AddPadding": AddPadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddPadding": "Add Padding"
}
