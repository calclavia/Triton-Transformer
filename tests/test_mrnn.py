import unittest

import torch
import torch.nn.functional as F
from ttx.mrnn.mrnn import mRNN
from ttx.mrnn.triton import mRNNFunction as mRNNFunctionTriton


class TestmRNN(unittest.TestCase):
    def test_causal_product_fwd_triton(self):
        torch.manual_seed(1)
        for bsz in range(1, 4):
            for dim in range(4, 128, 16):
                for l in range(4, 32, 8):
                    print('Testing:', bsz, l, dim)
                    module = mRNN(dim).to('cuda')

                    inputs = torch.randn(bsz, l, dim * 2, device='cuda')
                    state = torch.zeros(bsz, dim, device='cuda')

                    ref_output = module(inputs, state)

                    triton_output = mRNNFunctionTriton.apply(
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

    def test_causal_product_bwd_torch(self):
        torch.manual_seed(10)
        for bsz in range(1, 4):
            for dim in range(4, 128, 16):
                for l in range(4, 32, 8):
                    print('test_causal_product_bwd_torch testing:', bsz, l, dim)
                    module = mRNN(dim).to('cuda')

                    inputs = torch.randn(
                        bsz, l, dim * 2, device='cuda', requires_grad=True)
                    state = torch.zeros(
                        bsz, dim, device='cuda', requires_grad=True)

                    ref_output = module(inputs, state)
                    grad = torch.randn_like(ref_output)
                    ref_output.backward(grad)

                    ref_state_grad, ref_inputs_grad, ref_weight_grad = state.grad, inputs.grad, module.cell.weight.grad

                    triton_output = mRNNFunctionTriton.apply(
                        inputs, state, module.cell.weight)
                    triton_output.backward(grad)

                    state_grad, inputs_grad,  weight_grad = state.grad, inputs.grad, module.cell.weight.grad

                    try:
                        assert torch.allclose(
                            ref_state_grad, state_grad, atol=1e-2, rtol=1e-1)
                    except Exception as e:
                        print('state ref', ref_state_grad,
                              ref_state_grad.size())
                        print('state output', state_grad, state_grad.size())
                        raise e

                    try:
                        assert torch.allclose(
                            ref_inputs_grad, inputs_grad, atol=1e-2, rtol=1e-1)
                    except Exception as e:
                        print('inputs ref', ref_inputs_grad)
                        print('inputs output', inputs_grad)
                        raise e
                    try:
                        assert torch.allclose(
                            ref_weight_grad, weight_grad, atol=1e-2, rtol=1e-1)
                    except Exception as e:
                        print('weight_grad ref',  ref_weight_grad)
                        print('weight_grad output', weight_grad)
                        raise e


if __name__ == '__main__':
    unittest.main()
