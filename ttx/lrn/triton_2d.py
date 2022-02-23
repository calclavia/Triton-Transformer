import math

import triton
import triton.language as tl

import torch


@triton.jit
def lrn_kernel(
    q_ptr,  # [B, L, K]
    k_ptr,
    v_ptr,  # [B, L, V]
    z1_ptr,
    z2_ptr,
    state_ptr,  # [B, V, K]
    output_ptr,  # [B, L, V]
    batch, length, kdim, vdim,  # Size of the tensor dimensions
    K_BLOCK_SIZE: tl.constexpr,
    V_BLOCK_SIZE: tl.constexpr,
):
    # Paper: https://arxiv.org/pdf/2006.16236.pdf (Algorithm 1)
    # Reference: https://github.com/idiap/fast-transformers/blob/2fe048a14c2e67787f553e899123ca4ba9f27e76/fast_transformers/causal_product/causal_product_cpu.cpp#L82

    pid = tl.program_id(axis=0)

    # TODO: Pass in accumulator
    # Load state [V_BLOCK_SIZE, K_BLOCK_SIZE] for this batch
    state_offset = pid * kdim * vdim
    # state_v_offsets = tl.arange(0, V_BLOCK_SIZE)[:, None]
    # state_k_offsets = tl.arange(0, K_BLOCK_SIZE)[:, None]
    # state = tl.load(
    #     state_ptr + state_offset +
    #     state_v_offsets +
    #     state_k_offsets,
    #     mask=state_v_offsets < vdim & state_k_offsets < kdim, other=0
    # )
    state = tl.zeros((V_BLOCK_SIZE, K_BLOCK_SIZE), dtype=tl.float32)

    # Points to a single row
    kdim_ptrs = tl.arange(0, K_BLOCK_SIZE)
    vdim_ptrs = tl.arange(0, V_BLOCK_SIZE)
    k_mask = kdim_ptrs < kdim
    v_mask = kdim_ptrs < vdim

    # Offset for a single row in Q, K, V
    # Offset by batch size, seq len
    k_offsets = pid * length * kdim + kdim_ptrs
    v_offsets = pid * length * vdim + vdim_ptrs

    for _ in range(0, length, 1):

        # Load a single row of K and V as matrices.
        # [K_BLOCK_SIZE, 1]
        k = tl.load(k_ptr + k_offsets,
                    mask=k_mask, other=0)[:, None]
        # [1, V_BLOCK_SIZE]
        v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0)[None, :]

        # Compute context [V, 1] x [1, K] => [V, K]
        context = tl.dot(v, k)
        state += context

        # Load a single row of Q of shape [K, 1]
        # TODO: Loading this causes segfault
        # q = tl.load(q_ptr + k_offsets[:, None], mask=k_mask[:, None], other=0)

        # Compute output = S * Q. [V, K] x [K, 1] x  => [D, 1]
        # TODO: Correct equation
        # output = tl.dot(state, q)
        output = tl.dot(state, k)

        # TODO: Storing output causes IndexError: map::at
        tl.store(
            output_ptr + v_offsets[:, None],
            output,
            mask=v_mask[:, None]
        )

        # Move to next row
        k_offsets += kdim
        v_offsets += vdim


def lrn_fwd_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    state: torch.Tensor
):
    """
    All tensors are of shape [B, L, K]
    Except value, which is of [B, L, D]
    State is of shape [B, D, K]
    Output is: [B, L, D]
    """
    # We need to preallocate the output
    output = torch.empty_like(v)
    assert q.is_cuda and k.is_cuda and v.is_cuda and z1.is_cuda and z2.is_cuda and state.is_cuda
    assert q.is_contiguous and k.is_contiguous and v.is_contiguous and z1.is_contiguous and z2.is_contiguous and state.is_contiguous

    assert q.size() == k.size()
    assert q.size() == z1.size()
    assert q.size() == z2.size()

    kdim = k.size(-1)
    batch, length, vdim = output.size()
    print(batch, length, kdim, vdim)

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
    k_block_size = triton.next_power_of_2(kdim)
    v_block_size = triton.next_power_of_2(vdim)
    num_warps = min(max(k_block_size // 256, 1), 8)
    lrn_kernel[grid](
        q, k, v, z1, z2, state, output, batch, length, kdim, vdim,
        num_warps=num_warps,
        K_BLOCK_SIZE=k_block_size,
        V_BLOCK_SIZE=v_block_size
    )
    return output
