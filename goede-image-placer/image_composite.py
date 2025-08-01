import torch
from PIL import Image
import numpy as np

class ImageComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "subject_image": ("IMAGE",),
                "spacing": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "composite"

    CATEGORY = "Goede"

    def composite(self, background_image, subject_image, spacing):
        # Convert tensors to PIL images
        background_pil = Image.fromarray(np.clip(255. * background_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        subject_pil = Image.fromarray(np.clip(255. * subject_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        # Calculate the new size of the subject
        new_width = background_pil.width - 2 * spacing
        aspect_ratio = subject_pil.height / subject_pil.width
        new_height = int(new_width * aspect_ratio)

        # Resize the subject
        resized_subject = subject_pil.resize((new_width, new_height))

        # Create a new image with the background
        composite_image = Image.new("RGBA", background_pil.size)
        composite_image.paste(background_pil, (0, 0))

        # Calculate the position to paste the subject
        paste_x = spacing
        paste_y = (background_pil.height - new_height) // 2

        # Ensure subject is RGBA
        resized_subject = resized_subject.convert("RGBA")

        # Paste the subject onto the background
        composite_image.paste(resized_subject, (paste_x, paste_y), resized_subject)

        # Convert the composite image back to a tensor
        composite_tensor = torch.from_numpy(np.array(composite_image).astype(np.float32) / 255.0).unsqueeze(0)

        # Create a 3-channel version of the composite image
        composite_image_rgb = composite_image.convert("RGB")
        composite_tensor_rgb = torch.from_numpy(np.array(composite_image_rgb).astype(np.float32) / 255.0).unsqueeze(0)

        return (composite_tensor, composite_tensor_rgb)

NODE_CLASS_MAPPINGS = {
    "ImageComposite": ImageComposite
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageComposite": "Image Composite"
}
