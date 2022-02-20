import torch
import triton

from ttx.lrn.torch import lrn_torch, lrn_torch_1d
from ttx.lrn.triton_1d import LRN1DFunction


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        # different possible values for `x_name`
        x_vals=[i for i in range(8, 1024, 128)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['torch', 'triton', 'sru'],
        # label name for the lines
        line_names=["TorchScript", "Triton", 'SRU'],
        # line styles
        styles=[('green', '-'), ('red', '-'), ('blue', '-')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="forward-backward performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    device = 'cuda'
    bsz = 1
    dim = 256

    x = torch.randn(bsz, seq_len, dim,
                    device='cuda', requires_grad=True)
    z = torch.randn(bsz, seq_len, dim,
                    device='cuda', requires_grad=True)
    state = torch.randn(
        bsz, dim, device='cuda', requires_grad=True)
    grad = torch.randn(bsz, seq_len, dim, device='cuda')
    input_vars = (x, z, state)

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lrn_torch_1d(*input_vars).backward(grad))

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: LRN1DFunction.apply(*input_vars).backward(grad)
        )

    if provider == 'sru':
        from sru import SRUCell
        module = SRUCell(dim, dim).to('cuda')
        inputs = torch.randn(seq_len, bsz, dim, device='cuda')
        grad = torch.randn(seq_len, bsz, dim, device='cuda')
        U, V = module.compute_UV(
            inputs, None, None
        )
        c0 = torch.zeros(bsz, dim, dtype=inputs.dtype,
                         device=inputs.device, requires_grad=True)

        U = U.detach().requires_grad_()
        V = U.detach().requires_grad_()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: module.apply_recurrence(
                U, V, inputs, c0, None, None, None
            )[0].backward(grad))

    def perf(ms): return ms
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path='./out')
