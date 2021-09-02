import torch
import triton
from fast_transformers.causal_product import CausalDotProduct

from ttx.attention.causal_product import (causal_product_naive_cumsum,
                                          causal_product_triton)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['seq_len'],
        # different possible values for `x_name`
        x_vals=[i for i in range(8, 16, 64)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['torch', 'fast_transformers', 'triton'],
        # label name for the lines
        line_names=["PyTorch", 'fast_transformers', "Triton"],
        # line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="performance",
        args={},
    )
)
def benchmark(seq_len, provider):
    device = 'cuda'
    dim_size = 32
    size = (1, 1, seq_len, dim_size)
    vsize = (1, 1, seq_len, dim_size)

    scale_factor = dim_size ** -0.25
    q = torch.randn(*size, device=device).abs() * scale_factor
    k = torch.randn(*size, device=device).abs() * scale_factor
    v = torch.randn(*size, device=device)

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: causal_product_naive_cumsum(q, k, v))

    if provider == 'fast_transformers':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: CausalDotProduct.apply(q, k, v))

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_product_triton(
            q.view(-1, *size[2:]),
            k.view(-1, *size[2:]),
            v.view(-1, *vsize[2:])
        ))

    def perf(ms): return ms
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path='./out')
