import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@torch.jit.script
def lrn_torch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        state: torch.Tensor) -> torch.Tensor:
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    x = value.transpose(0, 1)
    for i in range(len(x)):
        # state is [B, D, K]. Z is [B, K, 1]. Key is [B, 1, K]
        state = state +\
            value[:, i].unsqueeze(-1) @ key[:, i].unsqueeze(-2)
        # state = state @ z1[:, i].unsqueeze(-1) @ z2[:, i].unsqueeze(-2) +\
        #     value[:, i].unsqueeze(-1) @ key[:, i].unsqueeze(-2)
        outputs += [(state @ query.unsqueeze(-1)).squeeze(-1)]
    outputs = torch.stack(outputs, dim=0)
    outputs = outputs.transpose(0, 1)
    return outputs


@torch.jit.script
def lrn_torch_1d(x: torch.Tensor, z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for i in range(x.size(1)):
        # state is [B, D]. Z is [B, D]. X is [B, D]
        state = state * z[:, i] + x[:, i]
        outputs += [state]
    outputs = torch.stack(outputs, dim=1)
    return outputs
