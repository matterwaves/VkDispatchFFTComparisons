import dataclasses

@dataclasses.dataclass
class Params:
    shape: tuple
    axis: int
    r2c: bool
    inverse: bool
    warmup: int
    file_name: str
    iter_count: int
    iter_batch: int

    def __init__(self,
                 fft_size: int,
                 batches_outer: int,
                 batches_inner: int,
                 r2c: bool,
                 inverse: bool,
                 warmup: int,
                 iter_count: int,
                 iter_batch: int):
        my_shape_list = [fft_size]
        my_axis = 0

        if batches_outer > 1:
            my_shape_list.insert(0, batches_outer)
            my_axis += 1
        
        if batches_inner > 1:
            my_shape_list.append(batches_inner)

            assert not r2c, "r2c is not supported for inner batches"

        self.shape = tuple(my_shape_list)
        self.axis = my_axis
        self.r2c = r2c
        self.inverse = inverse
        self.warmup = warmup
        self.iter_count = iter_count
        self.iter_batch = iter_batch

        assert iter_count % iter_batch == 0, "iter_count must be divisible by iter_batch"

        self.file_name = f"fft_{fft_size}_"
        self.file_name += f"batches_outer_{batches_outer}_"
        self.file_name += f"batches_inner_{batches_inner}_"
        self.file_name += f"r2c_{r2c}_"
        self.file_name += f"inverse_{inverse}_"
        self.file_name += f"iter_count_{iter_count}_"
        self.file_name += f"iter_batch_{iter_batch}_"
        self.file_name += f"warmup_{warmup}.txt"

def record_result(output_folder: str, params: Params, time: float):
    out_file = f"{output_folder}/{params.file_name}"

    with open(out_file, 'w') as f:
        f.write(str(time))

def params_from_filename(file_name: str) -> Params:
    parts = file_name.split('_')
    fft_size = int(parts[1])
    batches_outer = int(parts[4])
    batches_inner = int(parts[7])
    r2c = parts[9] == 'True'
    inverse = parts[11] == 'True'
    iter_count = int(parts[14])
    iter_batch = int(parts[17])  # Remove the .txt part
    warmup = int(parts[19].split('.')[0])  # Remove the .txt part

    return Params(fft_size, batches_outer, batches_inner, r2c, inverse, warmup, iter_count, iter_batch)