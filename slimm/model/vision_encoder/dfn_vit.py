import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, LayerNorm
from qwen_vl_utils import process_vision_info
from typing import Optional, List, Union, Tuple
from copy import deepcopy

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel

from transformers.activations import ACT2FN
from transformers.utils import logging

from .merger import MERGER_CLASSES

logger = logging.get_logger(__name__)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class Qwen2VLDFNVisionTransformer(Qwen2VisionTransformerPretrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.use_deepstack = getattr(config, 'use_deepstack', False)
        self.deepstack_type = getattr(config, 'deepstack_type', 'v1') if self.use_deepstack else None

        del self.merger

        kwargs = {}
        if self.use_deepstack:
            # FIXME: tmp impelmentation
            self.feature_idxs = [8, 16, 24, 30]
            MERGER_CLASS = MERGER_CLASSES['deepstack_{}'.format(self.deepstack_type)]
            self.merger = MERGER_CLASS(
                dim=config.llm_hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size, **kwargs
            )
        else:
            MERGER_CLASS = MERGER_CLASSES['default']
            self.merger = MERGER_CLASS(
                dim=config.llm_hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
            )


    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states_raw = hidden_states
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states_list = ()

        for i, blk in enumerate(self.blocks):
            if self.use_deepstack and not self.deepstack_type.startswith('efficient') and i in self.feature_idxs:
                hidden_states_list += (hidden_states,)

            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)


        args = []
        if self.deepstack_type is not None and self.deepstack_type.startswith('efficient'):
            args = [grid_thw]
        elif self.deepstack_type is not None:
            args = [hidden_states_list]

        return self.merger(hidden_states, *args)
