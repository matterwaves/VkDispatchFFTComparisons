import torch
import tqdm
import time
import test_params

def do_fft_instance(buffer: torch.Tensor, params: test_params.Params) -> None:
    if params.r2c:
        torch.fft.rfft(buffer, axis=params.axis) #, overwrite_x=True)
    else:
        torch.fft.fft(buffer, axis=params.axis) #, overwrite_x=True)

    if params.inverse:
        if params.r2c:
            torch.fft.irfft(buffer, axis=params.axis) #, overwrite_x=True)
        else:
            torch.fft.ifft(buffer, axis=params.axis) #, overwrite_x=True)

def do_fft_batch(buffer: torch.Tensor, params: test_params.Params) -> None:
    for _ in range(params.iter_batch):
        do_fft_instance(buffer, params)

def run_torch(params: test_params.Params) -> float:
    buffer = torch.empty(
        params.shape,
        dtype=torch.complex64 if not params.r2c else torch.float32,
        device='cuda'
    )
    
    for _ in range(params.warmup):
        do_fft_batch(buffer, params)

    torch.cuda.synchronize()

    status_bar = tqdm.tqdm(total=params.iter_count)
    
    start_time = time.time()

    for _ in range(params.iter_count // params.iter_batch):
        do_fft_batch(buffer, params)
        status_bar.update(params.iter_batch)

    torch.cuda.synchronize()

    elapsed_time = time.time() - start_time

    status_bar.close()

    return elapsed_time
