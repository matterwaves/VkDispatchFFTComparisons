#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status)); \
            return 1; \
        } \
    } while (0)

#define CHECK_CUFFT(call) \
    do { \
        cufftResult status = (call); \
        if (status != CUFFT_SUCCESS) { \
            fprintf(stderr, "cuFFT error: %d\n", status); \
            return 1; \
        } \
    } while (0)

int main() {
    const int n0 = 16384;      // First dimension
    const int n1 = 256;   // FFT length (middle axis)
    const int n2 = 64;   // Last dimension
    const int n_ffts = 1000;
    const int n_warmup = 100;

    // Total elements
    size_t n_elements = size_t(n0) * n1 * n2;
    size_t buf_size = n_elements * sizeof(cufftComplex);

    printf("Allocating %.2f GB device buffer...\n", buf_size / 1e9);
    cufftComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, buf_size));
    CHECK_CUDA(cudaMemset(d_data, 0, buf_size));
    int rank = 1;  
    int n[] = { n1 };           
    int istride = n2;            
    int idist = 1;             
    int batch = n0 * n2;      

    // This array was defined but not used correctly
    // int inembed[] = { n1, n2 };

    cufftHandle plan;
    CHECK_CUFFT(cufftPlanMany(
        &plan,
        rank, 
        n,
        NULL, istride, idist,  // Fixed formatting - removed extra NULL
        NULL, istride, idist,  // Fixed formatting - removed extra NULL
        CUFFT_C2C,
        batch
    ));

    // Warm-up
    for (int i = 0; i < n_warmup; ++i)
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_ffts; ++i)
        CHECK_CUFFT(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double buffers_per_sec = n_ffts / elapsed;

    printf("Processed %d buffers of size (%d, %d, %d) in %.4f seconds\n",
           n_ffts, n0, n1, n2, elapsed);
    printf("Throughput: %.2f buffers/sec of size (%d, %d, %d)\n",
           buffers_per_sec, n0, n1, n2);

    cufftDestroy(plan);
    cudaFree(d_data);
    return 0;
}
