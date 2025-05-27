import click

import test_params

def dispatch_test(backend: str, params: test_params.Params) -> float:
    if backend == "vkdispatch":
        import vkdispatch_backend
        return vkdispatch_backend.run_vkdispatch(params)
    elif backend == "cupy":
        import cupy_backend
        return cupy_backend.run_cupy(params)
    elif backend == "torch":
        import torch_backend
        return torch_backend.run_torch(params)
    else:
        raise ValueError(f"Unknown backend: {backend}")

@click.command()
@click.help_option("--help", "-h")
@click.option('--out_folder', help='Output folder')
@click.option('--backend', default=None, help='Select the backend to use (e.g., "vkdispatch", "cupy", or "torch").')
@click.option('--fft_size', help='Size of the FFT.', type=int)
@click.option('--batches_outer', help='Number of outer batches.', type=int)
@click.option('--batches_inner', help='Number of inner batches.', type=int)
@click.option('--r2c', is_flag=True)
@click.option('--inverse', is_flag=True)
@click.option('--warmup', help='Number of warmup iterations', default=10, type=int)
@click.option('--iter_count', help='Number of iterations to run', default=1000, type=int)
@click.option('--iter_batch', help='Number of iterations per batch', default=10, type=int)
def main(
        out_folder: str,
        backend: str,
        fft_size: int,
        batches_outer: int,
        batches_inner: int,
        r2c: bool,
        inverse: bool,
        warmup: int,
        iter_count: int,
        iter_batch: int):

    params = test_params.Params(fft_size, batches_outer, batches_inner, r2c, inverse, warmup, iter_count, iter_batch)
    print(f"Running test of shape {params.shape} on axis {params.axis} with r2c={params.r2c} using backend '{backend}'")
    
    time_taken = dispatch_test(backend, params)
    
    print(f"Time taken: {time_taken:.2f} seconds")
    test_params.record_result(out_folder, params, time_taken)
    print(f"Results saved to {params.file_name}")

if __name__ == "__main__":
    main()