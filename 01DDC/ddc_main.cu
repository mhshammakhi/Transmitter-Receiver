#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "../utils/utils.h"

// Kernel defined in DDS.cu
__global__ void DDC(const float * __restrict__ d_data_Re,
                    const float * __restrict__ d_data_Im,
                    float * __restrict__ d_DDC_Re,
                    float * __restrict__ d_DDC_Im,
                    const int dataLength,
                    const float frequency,
                    const float freq_init);

static constexpr int THREADS_PER_BLOCK = 256;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <output.bin> <frequency> [freq_init]\n"
                  << "  frequency : normalized carrier frequency (cycles/sample, e.g. 0.1)\n"
                  << "  freq_init : initial phase offset in radians (default 0.0)\n";
        return 1;
    }

    const std::string inFile    = argv[1];
    const std::string outFile   = argv[2];
    const float       frequency = std::stof(argv[3]);
    const float       freq_init = (argc >= 5) ? std::stof(argv[4]) : 0.0f;

    // Load interleaved IQ binary data into pinned host memory
    PinnedFloatVector h_re, h_im;
    readBinData(h_re, h_im, inFile);
    const int N = static_cast<int>(h_re.size());

    // Allocate device buffers
    float *d_re{}, *d_im{}, *d_out_re{}, *d_out_im{};
    cudaMalloc(&d_re,     N * sizeof(float));
    cudaMalloc(&d_im,     N * sizeof(float));
    cudaMalloc(&d_out_re, N * sizeof(float));
    cudaMalloc(&d_out_im, N * sizeof(float));

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
    DDC<<<gridSize, THREADS_PER_BLOCK>>>(d_re, d_im, d_out_re, d_out_im,
                                         N, frequency, freq_init);
    cudaEventRecord(stop_kernel);

    // --- D2H copy ---
    PinnedFloatVector h_out_re(N), h_out_im(N);
    cudaMemcpy(h_out_re.data(), d_out_re, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_im.data(), d_out_im, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_total);

    // Block until all work (including D2H) is done
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

    // Throughput in MSamples/s  =  N / (time_ms * 1e3)
    const double rate_kernel = N / (ms_kernel * 1e3);
    const double rate_total  = N / (ms_total  * 1e3);

    std::cout << "\n--- Timing (" << N << " samples) ---\n"
              << "Kernel only          : " << ms_kernel << " ms"
              << "  |  " << rate_kernel << " MSamples/s\n"
              << "H2D + kernel + D2H   : " << ms_total  << " ms"
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

    return 0;
}
