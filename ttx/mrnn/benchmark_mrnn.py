import torch
import triton

from ttx.mrnn.mrnn import mRNN, mrnn_fwd_triton


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
        plot_name="performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    device = 'cuda'
    bsz = 1
    dim = 256

    with torch.no_grad():
        inputs = torch.randn(bsz, seq_len, dim * 2, device='cuda')
        state = torch.zeros(bsz, dim, device='cuda')

        module = torch.jit.script(mRNN(dim)).to('cuda')
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: module(inputs, state))

        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: mrnn_fwd_triton(
                inputs, state, module.cell.weight))

        if provider == 'sru':
            from sru import SRUCell
            module = SRUCell(dim, dim).to('cuda')
            inputs = torch.randn(seq_len, bsz, dim, device='cuda')
            U, V = module.compute_UV(
                inputs, None, None
            )
            c0 = torch.zeros(bsz, dim, dtype=inputs.dtype,
                             device=inputs.device)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: module.apply_recurrence(
                    U, V, inputs, c0, None, None, None
                ))

    def perf(ms): return ms
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path='./out')
