from .image_composite import NODE_CLASS_MAPPINGS as image_composite_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_composite_display_name_mappings
from .drop_shadow import NODE_CLASS_MAPPINGS as drop_shadow_mappings, NODE_DISPLAY_NAME_MAPPINGS as drop_shadow_display_name_mappings
from .add_padding import NODE_CLASS_MAPPINGS as add_padding_mappings, NODE_DISPLAY_NAME_MAPPINGS as add_padding_display_name_mappings
from .image_selector import NODE_CLASS_MAPPINGS as image_selector_mappings, NODE_DISPLAY_NAME_MAPPINGS as image_selector_display_name_mappings


NODE_CLASS_MAPPINGS = {**image_composite_mappings, **drop_shadow_mappings, **add_padding_mappings, **image_selector_mappings}
NODE_DISPLAY_NAME_MAPPINGS = {**image_composite_display_name_mappings, **drop_shadow_display_name_mappings, **add_padding_display_name_mappings, **image_selector_display_name_mappings}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
