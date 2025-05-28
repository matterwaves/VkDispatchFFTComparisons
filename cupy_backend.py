import cupy as cp
import tqdm
import time
import test_params

def do_fft_instance(out_buff: cp.array, buffer: cp.array, params: test_params.Params) -> None:
    if params.r2c:
        cp.fft.rfft(buffer, axis=params.axis) #, overwrite_x=True)
    else:
        cp.fft.fft(buffer, axis=params.axis) #, overwrite_x=True)

    if params.inverse:
        if params.r2c:
            cp.fft.irfft(buffer, axis=params.axis) #, overwrite_x=True)
        else:
            cp.fft.ifft(buffer, axis=params.axis) #, overwrite_x=True)

def do_fft_batch(out_buff: cp.array, buffer: cp.array, params: test_params.Params) -> None:
    for _ in range(params.iter_batch):
        do_fft_instance(out_buff, buffer, params)

def run_cupy(params: test_params.Params) -> float:
    buffer = cp.empty(
        params.shape,
        dtype=cp.complex64 if not params.r2c else cp.float32)
    
    output_buffer = cp.empty_like(buffer)
    
    for _ in range(params.warmup):
        do_fft_batch(output_buffer, buffer, params)

    cp.cuda.Stream.null.synchronize()

    status_bar = tqdm.tqdm(total=params.iter_count)
    
    start_time = time.time()

    for _ in range(params.iter_count // params.iter_batch):
        do_fft_batch(output_buffer, buffer, params)
        status_bar.update(params.iter_batch)

    cp.cuda.Stream.null.synchronize()

    elapsed_time = time.time() - start_time

    status_bar.close()

    return elapsed_time
