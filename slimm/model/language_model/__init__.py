# from .dfn_vit import Qwen2VLDFNVisionTransformer
# from .siglip import SiglipVisionModel

from .qwen2vl import CustomQwen2


LANGUAGE_MODEL_CLASSES = {
    'qwen2_vl': CustomQwen2,
    'qwen2': CustomQwen2,
}
