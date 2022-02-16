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
        state = state + f * input
        return state


class mRNNLayer(nn.Module):
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


def mrnn_bwd_torch(inputs, states, weight, grad):
    """
    states: [B, L + 1 (first state is the initial state), D]
    """
    B = inputs.size(0)
    L = inputs.size(1)
    D = inputs.size(-1) // 2

    input_grad = torch.zeros_like(inputs)
    state_grad = torch.ones(B, D, device=input_grad.device)
    df_dx = 0

    # Highest to lower number
    for t in range(L - 1, -1, -1):
        prev_df_dx = df_dx

        x = inputs[:, t, :D]
        z = inputs[:, t, D:]
        s = states[:, t]

        logit = weight * s + z

        df_dx = torch.sigmoid(logit)
        input_grad[:, t, :D] = df_dx * grad[:, t]

        state_grad = (df_dx.pow(2) * weight * x *
                      torch.exp(logit) + 1)*grad[:, t]

    return input_grad, state_grad


# @triton.autotune(configs=[
#     triton.Config(meta={'BLOCK_SIZE': 32}, num_warps=2),
#     triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
#     triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
# ],
#     key=['dim']
# )
@triton.jit
def mrnn_kernel(
    x_ptr,  # [B, L, 2*D]
    state_ptr,  # [B, D]
    weight_ptr,  # [D] Transition matrix
    output_ptr,  # [B, L, D]
    batch, length, dim,  # Size of the tensor dimensions
    ** meta,  # Optional meta-parameters for the kernel
):
    # Size of dimensions to work with
    BLOCK_SIZE = meta['BLOCK_SIZE']

    batch_id = tl.program_id(axis=0)
    dim_id = tl.program_id(axis=1)

    dim_offset = dim_id * BLOCK_SIZE

    # Points to a single row
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    vec_mask = dim_ptrs < dim

    # 1xD state matrix
    state = tl.load(state_ptr + batch_id * dim + dim_offset +
                    dim_ptrs, mask=vec_mask, other=0)
    # Load weight matrix [D, D]
    W = tl.load(weight_ptr+dim_id*BLOCK_SIZE + dim_ptrs, mask=vec_mask)

    # Offset by batch size and dim ID
    x_pos = batch_id * length * (2*dim) + dim_offset
    out_pos = batch_id * length * (dim) + dim_offset

    for _ in range(0, length, 1):
        # Offset for a single row in x_ptr
        x_offsets = x_pos + dim_ptrs
        out_offsets = out_pos + dim_ptrs

        # Load single row [D]
        x = tl.load(x_ptr + x_offsets, mask=vec_mask, other=0)
        x_f = tl.load(x_ptr + dim + x_offsets, mask=vec_mask, other=0)

        # Apply state transition
        f = tl.sigmoid(x_f + state * W)
        state += f * x

        # Store the result of this row
        tl.store(
            output_ptr + out_offsets,
            state,
            mask=vec_mask
        )

        # Move to next row
        x_pos += dim*2
        out_pos += dim


@triton.jit
def mrnn_bwd_kernel(
    x_ptr,  # [B, L, 2*D]
    state_ptr,  # [B, D]
    weight_ptr,  # [D] Transition matrix
    output_ptr,  # [B, L, D]
    batch, length, dim,  # Size of the tensor dimensions
    ** meta,  # Optional meta-parameters for the kernel
):
    # Paper: https://arxiv.org/pdf/2006.16236.pdf (Algorithm 1)
    # Reference: https://github.com/idiap/fast-transformers/blob/2fe048a14c2e67787f553e899123ca4ba9f27e76/fast_transformers/causal_product/causal_product_cpu.cpp#L82
    # extract meta-parameters
    BLOCK_SIZE = meta['BLOCK_SIZE']

    pid = tl.program_id(axis=0)

    # Points to a single row
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    vec_mask = dim_ptrs < dim

    # 1xD state matrix
    state = tl.load(state_ptr + (pid * dim) + dim_ptrs, mask=vec_mask, other=0)
    # Load weight matrix [D, D]
    W = tl.load(weight_ptr + dim_ptrs, mask=vec_mask)

    # Offset by batch size
    x_pos = pid * length * (2*dim)
    out_pos = pid * length * (dim)

    for _ in range(0, length, 1):
        # Offset for a single row in x_ptr
        x_offsets = x_pos + dim_ptrs
        out_offsets = out_pos + dim_ptrs

        # Load single row [D]
        x = tl.load(x_ptr + x_offsets, mask=vec_mask, other=0)
        x_f = tl.load(x_ptr + dim + x_offsets, mask=vec_mask, other=0)

        # Apply state transition
        f = tl.sigmoid(x_f + state * W)
        state += f * x

        # Store the result of this row
        tl.store(
            output_ptr + out_offsets,
            state,
            mask=vec_mask
        )

        # Move to next row
        x_pos += dim*2
        out_pos += dim


def mrnn_fwd_triton(inputs: torch.Tensor, state: torch.Tensor, weight: torch.Tensor):
    # We need to preallocate the output
    output = torch.empty((inputs.size(0), inputs.size(
        1), inputs.size(2)//2), device=inputs.device)
    assert inputs.is_cuda and state.is_cuda and output.is_cuda and weight.is_cuda
    assert inputs.size(-1) == state.size(-1) * 2
    batch, length, dim = output.size()
    # print('Input dim', batch, length, dim, vdim)

    block_size = int(2 ** math.ceil(math.log2(dim)))
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks
    def grid(meta): return (batch, math.ceil(dim / block_size))
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments
    # print('launched kernel...')
    # print('block_size', block_size, 'num blocks', batch)
    # print('qk', q, k, torch.matmul(q, k.transpose(-1, -2)))
    # print('v', v)
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_kernel[grid](
        inputs, state, weight, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    # print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


def mrnn_bwd_triton(
    inputs: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
):
    # We need to preallocate the output
    output = torch.empty((inputs.size(0), inputs.size(
        1), inputs.size(2)//2), device=inputs.device)
    assert inputs.is_cuda and state.is_cuda and output.is_cuda and weight.is_cuda
    assert inputs.size(-1) == state.size(-1) * 2
    batch, length, dim = output.size()
    # print('Input dim', batch, length, dim, vdim)

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks
    # def grid(meta): return (triton.cdiv(batch, meta['BLOCK_SIZE']),)
    def grid(meta): return (batch,)
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments
    # print('launched kernel...')
    block_size = int(2 ** math.ceil(math.log2(dim)))
    # print('block_size', block_size, 'num blocks', batch)
    # print('qk', q, k, torch.matmul(q, k.transpose(-1, -2)))
    # print('v', v)
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_bwd_kernel[grid](
        inputs, state, weight, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    # print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
