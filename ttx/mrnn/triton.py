import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# @triton.autotune(configs=[
#     triton.Config(meta={'BLOCK_SIZE': 32}, num_warps=2),
#     triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
#     triton.Config(meta={'BLOCK_SIZE': 512}, num_warps=6),
#     triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
# ],
#     key=['dim']
# )


@triton.jit
def mrnn_fwd_kernel(
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
    # Load transition vector [D]
    W = tl.load(weight_ptr + dim_offset + dim_ptrs, mask=vec_mask, other=0)

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


def mrnn_fwd_triton(inputs: torch.Tensor, state: torch.Tensor, weight: torch.Tensor):
    assert inputs.is_contiguous()
    assert state.is_contiguous()
    assert weight.is_contiguous()
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
    def grid(meta): return (batch, math.ceil(dim / meta['BLOCK_SIZE']))
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments

    block_size = 1  # int(2 ** math.ceil(math.log2(dim)))
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_fwd_kernel[grid](
        inputs, state, weight, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    # print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


def mrnn_bwd_full(inputs, states, weight, grad):
    """
    states: [B, L + 1 (first state is the initial state), D]
    weight: [D]
    grad: Gradient of loss w.r.t. state
    """
    B = inputs.size(0)
    L = inputs.size(1)
    D = inputs.size(-1) // 2

    inputs_grad = torch.empty_like(inputs)
    weight_grad = torch.zeros_like(weight)

    # Precompute certain values
    x = inputs[..., :D]
    z = inputs[..., D:]
    logits = weight * states[:, :-1] + z
    sigs = torch.sigmoid(logits)
    sig_sqrds = sigs.pow(2)
    x_exp_neg_logits = x * torch.exp(-logits)

    state_grad_pre = sig_sqrds * weight * x_exp_neg_logits + 1

    # state_grad = torch.zeros(B, D, device=inputs_grad.device)
    # recurrent_grads = torch.empty(B, L, D, device=inputs_grad.device)

    # # Highest to lower timestep
    # for t in range(L - 1, -1, -1):
    #     recurrent_grads[:, t] = grad[:, t] + state_grad
    #     # Gradient of s
    #     state_grad = state_grad_pre[:, t] * recurrent_grads[:, t]

    recurrent_grads = mrnn_bwd_triton(
        grad.contiguous(),
        state_grad_pre,
    )

    state_grad = state_grad_pre[:, 0] * recurrent_grads[:, 0]

    # Gradient of x
    inputs_grad[..., :D] = sigs * recurrent_grads
    # Gradient of z
    inputs_grad[..., D:] = sig_sqrds * \
        x_exp_neg_logits * recurrent_grads
    # Gradient of v
    weight_grad += (sig_sqrds * states[:, :-1] *
                    x_exp_neg_logits * recurrent_grads).sum(dim=[0, 1])

    return inputs_grad, state_grad, weight_grad


@triton.jit
def mrnn_bwd_kernel(
    grad_ptr,  # [B, L, D]
    state_grad_pre_ptr,  # [B, L, D]
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

    # D state gradient accumlator
    # TODO: float16 would be faster?
    state_grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Offset by batch size and dim ID (start at last time step)
    pos = batch_id * length * dim + length * dim + dim_offset

    for _ in range(0, length, 1):
        # Move to prev time step
        pos -= dim

        # Offset for a single row in x_ptr
        offsets = pos + dim_ptrs

        # Load single row [D]
        grad_t = tl.load(grad_ptr + offsets, mask=vec_mask, other=0)
        state_grad_pre_t = tl.load(
            state_grad_pre_ptr + offsets, mask=vec_mask, other=0)

        # Compute gradient
        output = grad_t + state_grad
        state_grad = state_grad_pre_t * output

        # Store the result of this row
        tl.store(
            output_ptr + offsets,
            output,
            mask=vec_mask
        )


def mrnn_bwd_triton(
    grad: torch.Tensor,
    state_grad_pre: torch.Tensor,
):
    batch, length, dim = grad.size()
    assert grad.size() == state_grad_pre.size()
    assert grad.is_contiguous()
    assert state_grad_pre.is_contiguous()

    # We need to preallocate the output
    output = torch.zeros((batch, length, dim), device=grad.device)

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks
    def grid(meta): return (batch, math.ceil(dim / meta['BLOCK_SIZE']))
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments

    block_size = 1  # int(2 ** math.ceil(math.log2(dim)))
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_bwd_kernel[grid](
        grad, state_grad_pre, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    return output


class mRNNFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, state: torch.Tensor, weight: torch.Tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        outputs = mrnn_fwd_triton(inputs, state, weight)

        ctx.save_for_backward(
            inputs,
            torch.cat((state.unsqueeze(1), outputs), dim=1),
            weight
        )

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inputs, states, weight = ctx.saved_tensors
        return mrnn_bwd_full(inputs, states, weight, grad_output)
