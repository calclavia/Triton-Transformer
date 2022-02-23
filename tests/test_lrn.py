import unittest

import torch
import torch.nn.functional as F
from ttx.lrn.torch import lrn_torch, lrn_torch_1d
from ttx.lrn.triton_2d import lrn_fwd_triton
from ttx.lrn.triton_1d import LRN1DFunction

from ttx.mrnn.triton import mRNNFunction as mRNNFunctionTriton


class TestLRN(unittest.TestCase):
    def test_triton_1d(self):
        torch.manual_seed(1)

        for bsz in range(1, 16, 4):
            for seqlen in range(1, 16, 4):
                for dim in range(1, 512, 128):
                    print('test_triton_1d', bsz, seqlen, dim)
                    x = torch.randn(bsz, seqlen, dim,
                                    device='cuda')
                    z = torch.randn(bsz, seqlen, dim,
                                    device='cuda')
                    state = torch.randn(
                        bsz, dim, device='cuda', requires_grad=True)

                    input_vars = ((1-torch.sigmoid(z))*x,
                                  torch.sigmoid(z), state)
                    input_vars = tuple(v.detach() for v in input_vars)
                    for v in input_vars:
                        v.requires_grad_()

                    ref_output = lrn_torch_1d(*input_vars)
                    grad = torch.randn_like(ref_output)
                    ref_output.backward(grad)

                    ref_grads = tuple(v.grad for v in input_vars)

                    # Clear grad
                    for v in input_vars:
                        v.grad = None

                    triton_output = LRN1DFunction.apply(*input_vars)
                    triton_output.backward(grad)
                    triton_grads = tuple(v.grad for v in input_vars)

                    assert len(ref_grads) == len(triton_grads)

                    try:
                        assert torch.allclose(ref_output, triton_output, atol=1e-2, rtol=1e-1), (
                            f'The maximum difference between ref and triton is '
                            f'{torch.max(torch.abs(ref_output - triton_output))}'
                        )
                    except Exception as e:
                        print('ref_output', ref_output)
                        print('triton_output', triton_output)
                        raise e

                    # Double check to ensure we match mRNN
                    '''
                    with torch.no_grad():
                        try:
                            triton_output = mRNNFunctionTriton.apply(
                                x.detach(),
                                z.detach(),
                                torch.zeros(dim, device='cuda'), input_vars[2].detach())
                            assert torch.allclose(ref_output,
                                                  triton_output, atol=1e-2, rtol=1e-1), (
                                f'The maximum difference between ref and triton is '
                                f'{torch.max(torch.abs(ref_output - triton_output))}'
                            )
                        except Exception as e:
                            print('ref_output', ref_output)
                            print('triton_output', triton_output)
                            raise e
                    '''
                    for i, (ref, out) in enumerate(zip(ref_grads, triton_grads)):
                        try:
                            assert torch.allclose(ref, out, atol=1e-2, rtol=1e-1), (
                                f'The maximum difference between ref and out is '
                                f'{torch.max(torch.abs(ref - out))}'
                            )
                        except Exception as e:
                            print(f'ref grad {i} {ref}')
                            print(f'out grad {i} {out}')
                            raise e

    def test_triton_2d(self):
        torch.manual_seed(1)
        bsz = 1
        seqlen = 2
        kdim = 4
        vdim = 2
        print(bsz, seqlen, kdim, vdim)
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
