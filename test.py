import subprocess

def run_test(
        out_folder: str,
        backend: str,
        fft_size: int,
        batches_outer: int,
        batches_inner: int,
        r2c: bool = False,
        inverse: bool = False,
        warmup: int = 10,
        iter_count: int = 1000,
        iter_batch: int = 10):
    
    command = [
        'python3', 'entrypoint.py',
        '--out_folder', out_folder,
        '--backend', backend,
        '--fft_size', str(fft_size),
        '--batches_outer', str(batches_outer),
        '--batches_inner', str(batches_inner),
        #'--r2c' if r2c else '',
        #'--inverse' if inverse else '',
        '--warmup', str(warmup),
        '--iter_count', str(iter_count),
        '--iter_batch', str(iter_batch)
    ]

    if r2c:
        command.append('--r2c')
    if inverse:
        command.append('--inverse')

    print("Running command:", ' '.join(command))

    try:
        subprocess.run(command)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the test:")
        print(e.stderr)

if __name__ == "__main__":
    batches_outer = 4096
    batches_inner = 16

    fft_size = 64

    while fft_size <= 4096:
        run_test(
            out_folder='results',
            backend='vkdispatch',
            fft_size=fft_size,
            batches_outer=batches_outer * (4096 // fft_size),
            batches_inner=batches_inner
        )
        fft_size *= 2