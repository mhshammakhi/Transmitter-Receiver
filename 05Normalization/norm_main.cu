#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "../utils/utils.h"

// Normalization mode select:
//   0 = UnitPowerNormalization - scale so the average power (RMS^2) of the
//       block is 1
//   1 = AGC - scale so the peak amplitude (max |sample|) of the block is 1
#define USE_AGC 0

// Defined in kernel.cu
extern __device__ __managed__ float d_NormalizeFactor_MF;
extern __device__ __managed__ int d_nValid_vec[1];

__global__
void computePowerSamples(const float *d_data_Re, const float *d_data_Im, float *d_power,
                         const bool isOqpsk = false, const int dataLen = 0);
__global__
void parallelSum_arbitraryLen(float *a, float *sum, const bool isOqpsk = false, const int dataLen = 0);
__global__
void parallelMax_arbitraryLen(float *a, float *sum, const bool isOqpsk = false, const int dataLen = 0);
__global__
void NF_LoopFilter_MF(float *d_sum, const bool isOqpsk = false, const int dataLen = 0);
__global__
void NF_LoopFilter_AGC(float *d_max, const bool isOqpsk = false, const int dataLen = 0);
__global__
void Normalize_MF(float *d_data_Re, float *d_data_Im, float *d_normalize_Re, float *d_normalize_Im,
                  const bool isOqpsk = false, const int dataLen = 0);

// The arbitraryLen reduction kernels hard-code a block size of 1024.
static constexpr int THREADS_PER_BLOCK = 1024;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin>\n";
        return 1;
    }

    const std::string inFile  = argv[1];
    const std::string outFile = argv[2];

    // Load interleaved IQ binary data into pinned host memory
    PinnedFloatVector h_re, h_im;
    readBinData(h_re, h_im, inFile);
    const int N = static_cast<int>(h_re.size());

    d_nValid_vec[0] = N;

    // Allocate device buffers
    float *d_re{}, *d_im{}, *d_out_re{}, *d_out_im{}, *d_power{}, *d_reduce{};
    cudaMalloc(&d_re,     N * sizeof(float));
    cudaMalloc(&d_im,     N * sizeof(float));
    cudaMalloc(&d_out_re, N * sizeof(float));
    cudaMalloc(&d_out_im, N * sizeof(float));
    cudaMalloc(&d_power,  N * sizeof(float));
    cudaMalloc(&d_reduce, sizeof(float));
    cudaMemset(d_reduce, 0, sizeof(float));

    // Events: total spans H2D + kernel + D2H; kernel spans kernel only
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // --- H2D copy ---
    cudaEventRecord(start_total);
    cudaMemcpy(d_re, h_re.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im, h_im.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // --- Kernel ---
    const int gridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start_kernel);
    computePowerSamples<<<gridSize, THREADS_PER_BLOCK>>>(d_re, d_im, d_power);

#if USE_AGC
    parallelMax_arbitraryLen<<<gridSize, THREADS_PER_BLOCK>>>(d_power, d_reduce);
    NF_LoopFilter_AGC<<<1, 1>>>(d_reduce);
#else
    parallelSum_arbitraryLen<<<gridSize, THREADS_PER_BLOCK>>>(d_power, d_reduce);
    NF_LoopFilter_MF<<<1, 1>>>(d_reduce);
#endif

    Normalize_MF<<<gridSize, THREADS_PER_BLOCK>>>(d_re, d_im, d_out_re, d_out_im);
    cudaEventRecord(stop_kernel);

    // --- D2H copy ---
    PinnedFloatVector h_out_re(N), h_out_im(N);
    cudaMemcpy(h_out_re.data(), d_out_re, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_im.data(), d_out_im, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_total);

    cudaEventSynchronize(stop_total);

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // --- Timing results ---
    float ms_kernel{}, ms_total{};
    cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&ms_total,  start_total,  stop_total);

    const double rate_kernel = N / (ms_kernel * 1e3);
    const double rate_total  = N / (ms_total  * 1e3);

    std::cout << "\n--- Timing (" << N << " samples, mode="
#if USE_AGC
              << "AGC"
#else
              << "UnitPowerNormalization"
#endif
              << ") ---\n"
              << "Normalize factor      : " << d_NormalizeFactor_MF << "\n"
              << "Kernel only           : " << ms_kernel << " ms"
              << "  |  " << rate_kernel << " MSamples/s\n"
              << "H2D + kernel + D2H    : " << ms_total  << " ms"
              << "  |  " << rate_total  << " MSamples/s\n\n";

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    recordData(h_out_re.data(), h_out_im.data(), N, outFile);

    cudaFree(d_re);
    cudaFree(d_im);
    cudaFree(d_out_re);
    cudaFree(d_out_im);
    cudaFree(d_power);
    cudaFree(d_reduce);

    return 0;
}
