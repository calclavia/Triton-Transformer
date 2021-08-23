import unittest

import torch
from ttx.attention.causal_product import (causal_product_naive,
                                          causal_product_triton)
from ttx.attention.causal_product_bwd import causal_product_bwd_triton


class TestCausalDotProduct(unittest.TestCase):
    def test_fast_transformers(self):
        # Verifies the naive implementation against FastTransformers
        from fast_transformers.causal_product import CausalDotProduct
        torch.manual_seed(0)
        size = (1, 1, 5, 10)

        q = torch.rand(*size, device='cuda')
        k = torch.rand(*size, device='cuda')
        v = torch.rand(*size, device='cuda')

        ref_output = CausalDotProduct.apply(q, k, v).cpu()
        print('ref size', ref_output.size())
        naive_output = causal_product_naive(
            q, k, v).cpu()
        print('output size', naive_output.size())

        print(
            f'The maximum difference between ref and naive is '
            f'{torch.max(torch.abs(ref_output - naive_output))}'
        )
        assert torch.allclose(ref_output, naive_output)

    def test_triton(self):
        torch.manual_seed(0)
        size = (1, 1, 5, 3)

        q = torch.rand(*size, device='cuda')
        k = torch.rand(*size, device='cuda')
        v = torch.rand(*size, device='cuda')

        ref_output = causal_product_naive(q, k, v).cpu()
        triton_output = causal_product_triton(
            q.squeeze(1), k.squeeze(1), v.squeeze(1)).cpu()
        print('triton_output size', triton_output.size())

        print('ref_output', ref_output)
        print('triton_output', triton_output)

        # test_output = torch.matmul(torch.matmul(
        #     q, k.transpose(-1, -2)), v).squeeze(1).cpu()
        # print(torch.max(torch.abs(test_output - triton_output)))

        assert torch.allclose(ref_output, triton_output), (
            f'The maximum difference between ref and triton is '
            f'{torch.max(torch.abs(ref_output - triton_output))}'
        )

    def test_torch_causal_product_bwd(self):
        # Verifies the naive implementation against FastTransformers
        from fast_transformers.causal_product import CausalDotProduct

        torch.manual_seed(0)
        bsz = 2
        num_heads = 2
        length = 5
        dim_size = 8

        size = (bsz, num_heads, length, dim_size)
        vsize = (bsz, num_heads, length, 13)

        q = torch.randn(*size, device='cuda', requires_grad=True)
        k = torch.randn(*size, device='cuda', requires_grad=True)
        v = torch.randn(*vsize, device='cuda', requires_grad=True)

        qksize = (bsz, num_heads, length, dim_size)

        ref_output = CausalDotProduct.apply(q, k, v)
        # ref_output = causal_linear_attention_naive(q, k, v)
        ref_output.retain_grad()
        ref_output.mean().backward()

        # Reference gradients
        q_grad_ref, k_grad_ref, v_grad_ref = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        q_grad, k_grad, v_grad = causal_product_bwd_triton(
            q.view(-1, *qksize[2:]),
            k.view(-1, *qksize[2:]),
            v.view(-1, *vsize[2:]),
            ref_output.grad.view(-1, *vsize[2:])
        )
        q_grad = q_grad.view(*qksize)
        k_grad = k_grad.view(*qksize)
        v_grad = v_grad.view(*vsize)

        assert torch.allclose(q_grad_ref, q_grad,
                              atol=1e-1, rtol=1e-1), (q_grad_ref, q_grad)
        # assert torch.allclose(v_grad_ref, v_grad,
        #                       atol=1e-1, rtol=1e-1), (v_grad_ref, v_grad)
        # assert torch.allclose(k_grad_ref, k_grad,
        #                       atol=1e-1, rtol=1e-1), (k_grad_ref, k_grad)


if __name__ == '__main__':
    unittest.main()
