import vkdispatch as vd
import tqdm
import time
import test_params

def run_vkdispatch(params: test_params.Params) -> float:
    if params.r2c:
        buffer = vd.RFFTBuffer(params.shape)
        buffer_shape = buffer.real_shape
    else:
        buffer = vd.Buffer(params.shape, var_type=vd.complex64)
        buffer_shape = buffer.shape
    
    sync_buffer = vd.Buffer((10,), var_type=vd.float32)

    cmd_stream = vd.CommandStream()

    vd.fft.fft(
        buffer,
        buffer_shape=buffer_shape,
        cmd_stream=cmd_stream,
        inverse=params.inverse,
        axis=params.axis,
        r2c=params.r2c
    )

    for _ in range(params.warmup):
        cmd_stream.submit(params.iter_batch)

    sync_buffer.read()

    status_bar = tqdm.tqdm(total=params.iter_count)
    
    start_time = time.time()

    for _ in range(params.iter_count // params.iter_batch):
        cmd_stream.submit(params.iter_batch)
        status_bar.update(params.iter_batch)

    sync_buffer.read()

    elapsed_time = time.time() - start_time

    status_bar.close()

    return elapsed_time