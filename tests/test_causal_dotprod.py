import unittest

import torch
from ttx.attention.causal_product import (causal_product_naive,
                                          causal_product_triton)


class TestCausalDotProduct(unittest.TestCase):
    '''
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
    '''

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


if __name__ == '__main__':
    unittest.main()
