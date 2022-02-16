import unittest

import torch
import torch.nn.functional as F
from ttx.attention.mrnn import mRNNLayer, mrnn_fwd_triton


class TestmRNN(unittest.TestCase):
    def test_causal_product_fwd_triton(self):
        torch.manual_seed(1)
        for bsz in range(1, 4):
            for dim in range(4, 128, 16):
                for l in range(4, 32, 4):
                    print('Testing:', bsz, l, dim)
                    module = mRNNLayer(dim).to('cuda')

                    inputs = torch.randn(bsz, l, dim * 2, device='cuda')
                    state = torch.zeros(bsz, dim, device='cuda')

                    ref_output = module(inputs, state)

                    triton_output = mrnn_fwd_triton(
                        inputs, state, module.cell.weight)

                    try:
                        assert torch.allclose(ref_output, triton_output, atol=1e-2, rtol=1e-1), (
                            f'The maximum difference between ref and triton is '
                            f'{torch.max(torch.abs(ref_output - triton_output))}'
                        )
                    except Exception as e:
                        print('ref_output', ref_output)
                        print('triton_output', triton_output)
                        raise e


if __name__ == '__main__':
    unittest.main()
