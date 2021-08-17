import torch
import triton
import triton.language as tl


def causal_product_naive(q, k, v, chunk_size=1, eps=1e-6):
    # From: https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
    # inefficient causal linear attention, without cuda code, for reader's reference
    # chunk size determines the amount of parallelism
    last_context_cumsum = 0
    outs = []

    for q, k, v in zip(*map(lambda t: t.chunk(chunk_size, dim=-2), (q, k, v))):
        context = torch.einsum('...nd,...ne->...nde', k, v)
        context_cumsum = last_context_cumsum + context.cumsum(dim=-3)
        out = torch.einsum('...nde,...nd->...ne',
                           context_cumsum, q)

        last_context_cumsum = context_cumsum[..., -1:, :, :]
        outs.append(out)

    return torch.cat(outs, dim=-2)


@triton.jit
def causal_product_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,  # *Pointer* to output vector
    batch, length, dim,  # Size of the tensor dimensions
    **meta,  # Optional meta-parameters for the kernel
):
    # Paper: https://arxiv.org/pdf/2006.16236.pdf (Algorithm 1)
    # Reference: https://github.com/idiap/fast-transformers/blob/2fe048a14c2e67787f553e899123ca4ba9f27e76/fast_transformers/causal_product/causal_product_cpu.cpp#L82
    # extract meta-parameters
    BLOCK_SIZE = meta['BLOCK_SIZE']

    pid = tl.program_id(axis=0)
    matrix_size = length * dim
    batch_offset = pid * matrix_size

    # DxD matrix containing the current state
    # FP32 accumulation
    state = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    cur_pos = batch_offset

    # Points to a single row
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    mask = dim_ptrs < dim

    for t in range(0, length, 1):
        # Offset for a single row in Q, K, V
        row_offsets = cur_pos + dim_ptrs

        # Load a single row of K and V as matrices. Both are vectors of shape [dim]
        k = tl.load(k_ptr + row_offsets, mask=mask, other=0)
        v = tl.load(v_ptr + row_offsets, mask=mask, other=0)
        # Compute context [D, D] matrix from [D, 1] x [1, D]
        context = tl.dot(k[:, None], v[None, :])
        # state += context

        # Load a single row of Q of shape [dim]
        q = tl.load(q_ptr + row_offsets, mask=mask, other=0)

        # Compute output = QKV. [1, D] x [D, D] => [1, D]
        output = tl.dot(q[None, :], context)

        # Store the result of this row
        # tl.store(output_ptr + row_offsets, output, mask=mask)
        tl.store(output_ptr + row_offsets[None, :], output, mask=mask[None, :])

        # Move to next row
        cur_pos += dim


def causal_product_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Accepts 3 tensors Q, K, V of shape: [B, L, D]
    """
    # We need to preallocate the output
    output = torch.empty_like(v)
    assert q.is_cuda and k.is_cuda and v.is_cuda and output.is_cuda
    assert q.size() == k.size()
    assert v.size() == k.size()
    batch, length, dim = output.size()
    print('Input dim', batch, length, dim)

    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
    # In this case, we use a 1D grid where the size is the number of blocks
    # def grid(meta): return (triton.cdiv(batch, meta['BLOCK_SIZE']),)
    def grid(meta): return (batch,)
    # NOTE:
    #  - each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
    #  - don't forget to pass meta-parameters as keywords arguments
    print('launched kernel...')
    causal_product_kernel[grid](
        q, k, v, output, batch, length, dim,
        num_warps=1, BLOCK_SIZE=32)
    print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
