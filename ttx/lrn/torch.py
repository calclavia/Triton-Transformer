import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


def lrn_torch(query, key, value, z1, z2, state):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    x = value.transpose(0, 1)
    for i in range(len(x)):
        # state is [B, D, K]. Z is [B, K, 1]. Key is [B, 1, K]
        state = state +\
            value.unsqueeze(-1) @ key.unsqueeze(-2)
        # state = state @ z1.unsqueeze(-1) @ z2.unsqueeze(-2) +\
        #     value.unsqueeze(-1) @ key.unsqueeze(-2)
        outputs += [(state @ query.unsqueeze(-1)).squeeze(-1)]
    outputs = torch.stack(outputs, dim=0)
    outputs = outputs.transpose(0, 1)
    return outputs
