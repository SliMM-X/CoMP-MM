from .dfn_vit import Qwen2VLDFNVisionTransformer
from .comp_siglip import CoMPSiglipVisionModel
from .comp_dinov2 import CoMPDinov2Model

VISION_TRANSFORMER_CLASSES = {
    'qwen2_vl': Qwen2VLDFNVisionTransformer,
    'siglip': CoMPSiglipVisionModel,
    'dinov2': CoMPDinov2Model,
}
