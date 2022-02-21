

import math
import triton.language as tl
import triton
import torch


# @triton.autotune(configs=[
#     triton.Config({'BLOCK_SIZE': 2**i}, num_warps=max(i // 2, 1))
#     for i in range(4, 10, 2)
# ],
#     key=['dim']
# )
@triton.jit
def lrn_1d_kernel(
    x_ptr,  # [B, L, D]
    z_ptr,  # [B, L, D]
    state_ptr,  # [B, D]
    output_ptr,  # *Pointer* to output vector
    batch, length, dim,  # Size of the tensor dimensions
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(axis=0)

    # Offset by batch size
    offset = batch_id * length * dim

    # Points to a sub dimension to operate on
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    vec_mask = dim_ptrs < dim

    # [D] state matrix
    state = tl.load(state_ptr + batch_id * dim +
                    dim_ptrs, mask=vec_mask, other=0)

    # Offset by batch size and dim ID
    pos = batch_id * length * dim

    for _ in range(0, length, 1):
        # Offset for a single row in x_ptr
        offsets = pos + dim_ptrs

        # Load single row [D]
        x = tl.load(x_ptr + offsets, mask=vec_mask, other=0)
        z = tl.load(z_ptr + offsets, mask=vec_mask, other=0)

        # Apply state transition
        state = state * z + x

        # Store the result of this row
        tl.store(
            output_ptr + offsets,
            state,
            mask=vec_mask
        )

        # Move to next row
        pos += dim


def lrn_fwd_triton_1d(
    x: torch.Tensor,
    z: torch.Tensor,
    state: torch.Tensor
):
    """
    All tensors are of shape [B, L, D]
    State is of shape [B, D]
    Output is: [B, L, D]
    """
    # We need to preallocate the output
    output = torch.empty_like(x)
    assert x.is_cuda and z.is_cuda and state.is_cuda
    assert x.is_contiguous and z.is_contiguous and state.is_contiguous

    assert x.size() == z.size()

    batch, length, dim = x.size()

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
    block_size = triton.next_power_of_2(dim)
    num_warps = min(max(block_size // 256, 1), 8)
    lrn_1d_kernel[grid](
        x, z, state, output, batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    return output


@triton.jit
def lrn_bwd_kernel(
    grad_ptr,  # [B, L, D]
    state_grad_pre_ptr,  # [B, L, D]
    output_ptr,  # [B, L, D]
    batch, length, dim,  # Size of the tensor dimensions
    BLOCK_SIZE: tl.constexpr,  # Optional meta-parameters for the kernel
):
    batch_id = tl.program_id(axis=0)
    dim_id = tl.program_id(axis=1)

    dim_offset = dim_id * BLOCK_SIZE

    # Points to a single row
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    vec_mask = dim_ptrs < dim

    # D state gradient accumlator
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


def lrn_bwd_triton(
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
    lrn_bwd_kernel[grid](
        grad, state_grad_pre, output,
        batch, length, dim,
        num_warps=num_warps,
        BLOCK_SIZE=block_size
    )
    return output


class LRN1DFunction(torch.autograd.Function):
    """
    Wolfram:
    derivative of f(x,z,s)=s*z+x
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, z: torch.Tensor, state: torch.Tensor):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        outputs = lrn_fwd_triton_1d(x, z, state)

        ctx.save_for_backward(
            x, z,
            # All the states (except the final state)
            torch.cat((state.unsqueeze(1), outputs[:, :-1]), dim=1),
        )

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x, z, states = ctx.saved_tensors

        B, L, D = x.size()

        recurrent_grads = lrn_bwd_triton(
            grad_output.contiguous(),
            z,
        )

        return recurrent_grads, states * recurrent_grads, z[:, 0] * recurrent_grads[:, 0]
