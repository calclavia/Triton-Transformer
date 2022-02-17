import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


class mRNNCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input, input_forget, state):
        # type: (Tensor, Tensor, Tensor) -> Tensor

        # For matmul version: torch.mm(state, self.weight.t())
        f = torch.sigmoid(
            input_forget + state * self.weight)
        # state = forget_gate * state + (1-forget_gate) * input
        state = f * state + (1-f) * input
        return state


class mRNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = mRNNCell(hidden_size)

    def forward(self, x, state):
        # type: (Tensor, Tensor) -> Tensor
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        x = x.transpose(0, 1)
        for i in range(len(x)):
            state = self.cell(
                x[i][:, :self.hidden_size],
                x[i][:, self.hidden_size:],
                state
            )
            outputs += [state]
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.transpose(0, 1)
        return outputs
