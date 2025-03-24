import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Union

class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class DeepStackPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        self.ln_q_deepstack = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_deepstack = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, dim),
        )


    def forward(self, x: torch.Tensor, x_list: List[torch.Tensor] = None) -> torch.Tensor:
        # return self.merger(x), self.merger_deepstack(x)
        x_out = self.mlp(self.ln_q(x).view(-1, self.hidden_size)) #(l,d)
        x_deepstack = self.mlp_deepstack(self.ln_q(x))  #(l*4,d)
        return x_out, x_deepstack


class DeepStackMidLayerMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, feat_reverse: bool = False) -> None:
        super().__init__()
        self.feat_reverse = feat_reverse
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        # FIXME: abs number
        num_stack = 4
        self.ln_q_deepstack_list = nn.ModuleList(nn.LayerNorm(context_dim, eps=1e-6) for _ in range(num_stack)) 
        self.mlp_deepstack_list = nn.ModuleList(nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        ) for _ in range(num_stack))

    def forward(self, x: torch.Tensor, x_list: List[torch.Tensor] = None) -> torch.Tensor:
        # return self.merger(x), self.merger_deepstack(x)
        x_out = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        x_deepstack = []
        if self.feat_reverse:
            x_list = x_list[::-1]

        for x_this, mlp, ln_q in zip(x_list, self.mlp_deepstack_list, self.ln_q_deepstack_list):
            x_deepstack.append(mlp(ln_q(x_this).view(-1, self.hidden_size)))

        x_deepstack = torch.stack(x_deepstack, dim=1).flatten(0,1)

        return x_out, x_deepstack


class DeepStackEfficientMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2, feat_reverse: bool = False) -> None:
        super().__init__()
        self.feat_reverse = feat_reverse
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        dim_hidden = self.hidden_size

        self.ln_q_deepstack = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_deepstack = nn.Sequential(
            nn.Linear(self.hidden_size*4, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim),
        )


    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x_deepstack = self.mlp(self.ln_q(x).view(-1, self.hidden_size))

        # FIXME
        h_stack, w_stack = 2, 2

        seq_lens = grid_thw[:,1] * grid_thw[:,2] //4
        x_out = x.view(-1, self.hidden_size)
        # x_out = x_out.split(seq_lens.tolist())
        x_out_idxs = torch.arange(x_out.shape[0], device=x_out.device)
        x_out_idxs = x_out_idxs.split(seq_lens.tolist())
        grid_thw_out = grid_thw.clone()
        grid_thw_out[:,1:] = grid_thw_out[:,1:] //2

        x_out_idxs = [xx.view(t, h//h_stack, h_stack, w//w_stack, w_stack).permute(0,1,3,2,4).flatten() for (xx, (t,h,w)) in zip(x_out_idxs, grid_thw_out)]
        x_out_idxs = torch.cat(x_out_idxs)

        x_out = x_out[x_out_idxs].contiguous()

        spatial_merge_size = self.spatial_merge_size
        x_out = x_out.view(-1, spatial_merge_size**2, x.shape[-1])
        x_out = self.mlp_deepstack(self.ln_q_deepstack(x_out).view(-1, self.hidden_size*4))

        return x_out, x_deepstack


class DeepStackDenseConnector(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        # FIXME: this is for fast implementation
        self.hidden_size = context_dim * (spatial_merge_size**2) * 5
        self.hidden_size_singlescale = context_dim * (spatial_merge_size**2)
        self.ln_q_x = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp_x = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor, x_list: List[torch.Tensor] = None) -> torch.Tensor:

        x_out = torch.stack([x, *x_list], dim=1)

        x_out = self.mlp_x(self.ln_q_x(x_out).view(-1, self.hidden_size))

        x_deepstack = x_out.new_zeros(x_out.shape[0]*4, x_out.shape[1])

        return x_out, x_deepstack


MERGER_CLASSES = {
    'default': PatchMerger,
    'deepstack': DeepStackPatchMerger,
    'deepstack_v1': DeepStackPatchMerger,
    'deepstack_midlayers': DeepStackMidLayerMerger,
    'deepstack_efficient': DeepStackEfficientMerger,
    'dense_connector': DeepStackDenseConnector
}
