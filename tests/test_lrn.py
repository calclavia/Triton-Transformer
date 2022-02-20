import unittest

import torch
import torch.nn.functional as F
from ttx.lrn.torch import lrn_torch
from ttx.lrn.triton import lrn_fwd_triton


class TestLRN(unittest.TestCase):
    def test_triton(self):
        torch.manual_seed(1)
        bsz = 1
        seqlen = 4
        kdim = 4
        vdim = 4

        q = torch.randn(bsz, seqlen, kdim, device='cuda')
        k = torch.randn(bsz, seqlen, kdim, device='cuda')
        v = torch.randn(bsz, seqlen, vdim, device='cuda')
        z1 = torch.randn(bsz, seqlen, kdim, device='cuda')
        z2 = torch.randn(bsz, seqlen, kdim, device='cuda')
        state = torch.randn(bsz, vdim, kdim, device='cuda')

        ref_output = lrn_torch(q, k, v, z1, z2, state)
        triton_output = lrn_fwd_triton(q, k, v, z1, z2, state)
        print('v', v)

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
