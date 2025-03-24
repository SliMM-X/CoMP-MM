
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig


logger = logging.get_logger(__name__)


class SliMMConfig(Qwen2VLConfig):
    model_type = "slimm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *args,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = Qwen2VLVisionConfig(**vision_config)
        elif vision_config is None:
            if hasattr(self, 'vision_config', None) is not None:
                del self.vision_config

