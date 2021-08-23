
import math
import torch
import triton
import triton.language as tl


@torch.no_grad()
@torch.jit.script
def causal_product_bwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor
):
    """
    Pytorch implementation of algorithm.
    O(1) memory, O(T) time.
    """
    batch, length, dim = q.size()
    batch, length, vdim = v.size()
    # Paper: https://arxiv.org/pdf/2006.16236.pdf (Algorithm 1)
    grad_Q = torch.zeros_like(q)
    grad_K = torch.zeros_like(k)
    grad_V = torch.zeros_like(v)

    state = torch.zeros(batch, dim, vdim, dtype=q.dtype, device=q.device)
    for t in range(length):
        state += torch.matmul(k[:, t, :, None], v[:, t, None, :])
        grad_Q[:, t] = torch.matmul(state, grad_out[:, t, :, None]).squeeze(-1)

    state = torch.zeros(batch, dim, vdim, dtype=q.dtype, device=q.device)
    for t in range(length - 1, -1, -1):
        state += torch.matmul(q[:, t, :, None], grad_out[:, t, None, :])
        grad_V[:, t] = torch.matmul(k[:, t, None, :], state).squeeze(1)
        grad_K[:, t] = torch.matmul(state, v[:, t, :, None]).squeeze(-1)

    return grad_Q, grad_K, grad_V


@triton.jit
def causal_product_bwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    grad_out,
    # *Pointer* to output vectors
    grad_Q_ptr,
    grad_K_ptr,
    grad_V_ptr,
    batch, length, dim, vdim,  # Size of the tensor dimensions
    **meta,  # Optional meta-parameters for the kernel
):
    # Paper: https://arxiv.org/pdf/2006.16236.pdf (Algorithm 1)
    # Reference: https://github.com/idiap/fast-transformers/blob/2fe048a14c2e67787f553e899123ca4ba9f27e76/fast_transformers/causal_product/causal_product_cpu.cpp#L82
    BLOCK_SIZE = meta['BLOCK_SIZE']

    pid = tl.program_id(axis=0)
    matrix_size = length * dim
    batch_offset = pid * matrix_size

    # DxM matrix containing the current state [D, M] matrix
    # FP32 accumulation
    state = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    cur_qk_pos = batch_offset
    cur_v_pos = batch_offset

    # Points to a single row
    dim_ptrs = tl.arange(0, BLOCK_SIZE)
    qkmask = dim_ptrs < dim
    vmask = dim_ptrs < vdim

    # Compute gradient for Q
    # Equation 13
    for _ in range(0, length, 1):
        # Offset for a single row in Q, K, V
        qk_row_offsets = cur_qk_pos + dim_ptrs
        v_row_offsets = cur_v_pos + dim_ptrs

        # Load the current row of K, V vectors.
        k = tl.load(k_ptr + qk_row_offsets, mask=qkmask, other=0)
        v = tl.load(v_ptr + v_row_offsets, mask=vmask, other=0)
        # Compute context [D, M] matrix from [D, 1] x [1, M]
        context = tl.dot(k[:, None], v[None, :])
        state += context

        # Load gradient
        g = tl.load(grad_out + v_row_offsets, mask=vmask, other=0)
        grad_q = tl.dot(state, g[:, None])

        # Store the result of this row
        tl.store(grad_Q_ptr + qk_row_offsets[:,
                 None], grad_q, mask=qkmask[:, None])

        # Move to next row
        cur_qk_pos += dim
        cur_v_pos += vdim

    # Compute gradient for K and V
    # Equation 14, 15
    # Reset state [D, M] matrix
    '''
    state *= 0

    for _ in range(0, length, 1):
        # Move back one row
        cur_pos -= dim

        # Offset for a single row in Q, K, V
        row_offsets = cur_pos + dim_ptrs

        # Load the current row of Q, K, V vectors. All are vectors of shape [dim]
        q = tl.load(q_ptr + row_offsets, mask=mask, other=0)
        k = tl.load(k_ptr + row_offsets, mask=mask, other=0)
        v = tl.load(v_ptr + row_offsets, mask=vmask, other=0)
        # Load gradient
        g = tl.load(grad_out + row_offsets, mask=vmask, other=0)
        # Compute context [D, M] matrix from [D, 1] x [1, M]
        context = tl.dot(q[:, None], g[None, :])
        # state += context

        # Compute gradients [1, D] x [D, M] => [1, M]
        grad_v = tl.dot(k[None, :], context)
        grad_v = tl.reshape(grad_v, (meta['BLOCK_SIZE'],))
        # grad_v = tl.dot(k[None, :], state)

        # Enabling the follownig leads to a hang

        # grad_k = tl.dot(state, v[:, None])
        # print(grad_v.shape)
        # print(grad_k.shape)
        # Store the result of this row
        # tl.store(grad_V_ptr + row_offsets[None,
        #          :], grad_v, mask=vmask[None, :])
        tl.store(grad_V_ptr + row_offsets, grad_v, mask=vmask)
        # tl.store(grad_K_ptr + row_offsets[:, None], grad_k, mask=mask[:, None])
    '''


def causal_product_bwd_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_out: torch.Tensor):
    """
    Accepts tensors Q, K, V of shape: [B, L, D]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and grad_out.is_cuda
    assert q.size() == k.size()
    assert q.size()[:-1] == v.size()[:-1], (q.size(), v.size())
    assert v.size() == grad_out.size()

    batch, length, dim = q.size()
    vdim = v.size(-1)

    # Allocate memory for the gradients
    grad_Q = torch.zeros_like(q)
    grad_K = torch.zeros_like(k)
    grad_V = torch.zeros_like(v)
    print('Input dim', batch, length, dim, vdim)

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
    block_size = int(2 ** math.ceil(math.log2(max(dim, vdim))))
    print('block_size', block_size, 'num blocks', batch)
    causal_product_bwd_kernel[grid](
        q, k, v, grad_out, grad_Q, grad_K, grad_V, batch, length, dim, vdim,
        num_warps=1,
        BLOCK_SIZE=block_size
    )
    print('waiting on output...')
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return grad_Q, grad_K, grad_V
