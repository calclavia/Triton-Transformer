import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# @triton.autotune(configs=[
#     triton.Config(meta={'BLOCK_SIZE': 2**i}, num_warps=max(i // 2, 1))
#     for i in range(4, 10, 2)
# ],
#     key=['dim']
# )
@triton.jit
def mrnn_fwd_kernel(
    x_ptr,  # [B, L, D]
    z_ptr,  # [B, L, D]
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

    # Points to a sub dimension to operate on
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    vec_mask = dim_ptrs < dim

    # 1xD state matrix
    state = tl.load(state_ptr + batch_id * dim + dim_offset +
                    dim_ptrs, mask=vec_mask, other=0)
    # Load transition vector [D]
    W = tl.load(weight_ptr + dim_offset + dim_ptrs, mask=vec_mask, other=0)

    # Offset by batch size and dim ID
    pos = batch_id * length * dim + dim_offset

    for _ in range(0, length, 1):
        # Offset for a single row in x_ptr
        offsets = pos + dim_ptrs

        # Load single row [D]
        x = tl.load(x_ptr + offsets, mask=vec_mask, other=0)
        z = tl.load(z_ptr + offsets, mask=vec_mask, other=0)

        # Apply state transition
        f = tl.sigmoid(z + state * W)
        state = f * state + (1 - f) * x

        # Store the result of this row
        tl.store(
            output_ptr + offsets,
            state,
            mask=vec_mask
        )

        # Move to next row
        pos += dim


def mrnn_fwd_triton(
        x: torch.Tensor,
        z: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor):
    assert x.is_contiguous()
    assert z.is_contiguous()
    assert state.is_contiguous()
    assert weight.is_contiguous()
    # We need to preallocate the output
    output = torch.empty_like(x)
    assert x.is_cuda and z.is_cuda and state.is_cuda and output.is_cuda and weight.is_cuda
    assert x.size(-1) == state.size(-1)
    batch, length, dim = output.size()
    # print('Input dim', batch, length, dim, vdim)

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    def grid(meta): return (batch, triton.cdiv(dim, meta['BLOCK_SIZE']))
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments

    block_size = triton.next_power_of_2(dim)
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_fwd_kernel[grid](
        x, z, state, weight, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    # print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# @triton.autotune(configs=[
#     triton.Config(meta={'BLOCK_SIZE': 2**i}, num_warps=max(i // 2, 1))
#     for i in range(4, 10, 2)
# ],
#     key=['dim']
# )
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
    # TODO: Is FP16 safe here?
    state_grad = tl.zeros((BLOCK_SIZE,), dtype=tl.float16)

    # Offset by batch size and dim ID (start at last time step)
    pos = batch_id * length * dim + length * dim + dim_offset

    for _ in range(0, length, 1):
        # Move to prev time step
        pos -= dim

        # Offset for a single row in x_ptr
        offsets = pos + dim_ptrs

        # Load single row [D]
        grad_t = tl.load(grad_ptr + offsets, mask=vec_mask,
                         other=0).to(tl.float16)
        state_grad_pre_t = tl.load(
            state_grad_pre_ptr + offsets, mask=vec_mask, other=0).to(tl.float16)

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
    output = torch.empty((batch, length, dim),
                         device=grad.device, dtype=torch.half)

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks
    def grid(meta): return (batch, math.ceil(dim / meta['BLOCK_SIZE']))
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments

    block_size = triton.next_power_of_2(dim)
    num_warps = min(max(block_size // 256, 1), 8)
    mrnn_bwd_kernel[grid](
        grad, state_grad_pre, output,
        batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    return output


class mRNNFunction(torch.autograd.Function):
    """
    Wolfram:
    derivative of f(x,z,w,s)=sigmoid(z+w*s)*s + (1-sigmoid(z+w*s)) * x
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, state: torch.Tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        outputs = mrnn_fwd_triton(x, z, state, weight)

        # TODO: Storing all these states leads to O(b*l*d) memory
        ctx.save_for_backward(
            x, z,
            # All the states (except the final state)
            torch.cat((state.unsqueeze(1), outputs[:, :-1]), dim=1),
            weight
        )

        return outputs

    @staticmethod
    @torch.jit.script
    def precompute_grad_components(x, z, states, weight):
        D = x.size(-1)
        # Precompute certain values
        states_minus_x = states - x
        logits = weight * states + z
        exp_logits = torch.exp(logits)

        sigs = 1 / (1 + exp_logits)
        sig_sqrds = sigs.pow(2)
        state_grad_pre = sig_sqrds * exp_logits * \
            (exp_logits + states_minus_x * weight + 1)
        return state_grad_pre, (states, sigs, sig_sqrds, exp_logits, states_minus_x)

    @staticmethod
    @torch.jit.script
    def post_grad_components(
        state_grad_pre,
        recurrent_grads,
        states, sigs, sig_sqrds, exp_logits, states_minus_x
    ):
        state_grad = state_grad_pre[:, 0] * recurrent_grads[:, 0]
        x_grad = sigs * recurrent_grads
        z_grad = sig_sqrds * exp_logits * states_minus_x * recurrent_grads
        # Gradient of v
        weight_grad = (z_grad * states).sum(dim=[0, 1])
        return x_grad, z_grad, weight_grad, state_grad

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, z, states, weight = ctx.saved_tensors

        """
        states: [B, L + 1 (first state is the initial state), D]
        weight: [D]
        grad: Gradient of loss w.r.t. state
        """
        B = x.size(0)
        L = x.size(1)
        D = x.size(-1) // 2

        state_grad_pre, args = mRNNFunction.precompute_grad_components(
            x, z, states, weight
        )

        # Torch version
        # state_grad = torch.zeros(B, D, device=inputs_grad.device)
        # recurrent_grads = torch.empty(B, L, D, device=inputs_grad.device)

        # # Highest to lower timestep
        # for t in range(L - 1, -1, -1):
        #     recurrent_grads[:, t] = grad[:, t] + state_grad
        #     # Gradient of s
        #     state_grad = state_grad_pre[:, t] * recurrent_grads[:, t]

        recurrent_grads = mrnn_bwd_triton(
            grad_output.contiguous(),
            state_grad_pre,
        )

        return mRNNFunction.post_grad_components(
            state_grad_pre,
            recurrent_grads,
            *args
        )
