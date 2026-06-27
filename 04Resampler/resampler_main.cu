#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "../utils/utils.h"

// Defined in kernel.cu
void resamplerComputeStep_Host(const int &num_threads, const int &num_blocks,
                              const float &iFs, const float &oFs);

__global__
void resampler(float *out_re, float *out_im, int *outLen, float *in_re, float *in_im,
               const float iFs, const float oFs, const int frameLen);

// in_re/in_im reserve this many slots ahead of the real samples for the
// cubic interpolator's tap history. On a cold start (d_resample_nFromPrev==0,
// d_last_startInterpIndex==1, the kernel.cu defaults) this region is never
// read, but it is zeroed anyway since cudaMalloc leaves it uninitialized.
static constexpr int HISTORY_LEN = 4;
static constexpr int THREADS_PER_BLOCK = 256;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin> <iFs> <oFs>\n"
                  << "  iFs : input sample rate\n"
                  << "  oFs : output sample rate\n";
        return 1;
    }

    const std::string inFile  = argv[1];
    const std::string outFile = argv[2];
    const float       iFs     = std::stof(argv[3]);
    const float       oFs     = std::stof(argv[4]);

    // Load interleaved IQ binary data into pinned host memory
    PinnedFloatVector h_re, h_im;
    readBinData(h_re, h_im, inFile);
    const int frameLen = static_cast<int>(h_re.size());

    // Worst-case output length for a single full-buffer, cold-start pass,
    // mirroring the outlen formula computed inside the resampler kernel
    // (lastIndex = frameLen-1, startInterpIndex = 1, since nFromPrev == 0).
    const double decimationRate = static_cast<double>(iFs) / oFs;
    const double lastIndex      = frameLen - 1;
    const double startInterpIdx = 1.0;
    const double temp = std::floor((lastIndex - startInterpIdx) / decimationRate) * decimationRate + startInterpIdx;
    const int outputCapacity = static_cast<int>(std::floor((lastIndex - startInterpIdx) / decimationRate))
                              + (temp < lastIndex - 1 ? 1 : 0)
                              + 4; // margin for float-vs-double rounding at the boundary

    // Allocate device buffers
    float *d_re{}, *d_im{}, *d_out_re{}, *d_out_im{};
    int *d_outLen{};
    cudaMalloc(&d_re,     (frameLen + HISTORY_LEN) * sizeof(float));
    cudaMalloc(&d_im,     (frameLen + HISTORY_LEN) * sizeof(float));
    cudaMalloc(&d_out_re, outputCapacity * sizeof(float));
    cudaMalloc(&d_out_im, outputCapacity * sizeof(float));
    cudaMalloc(&d_outLen, sizeof(int));
    cudaMemset(d_re, 0, HISTORY_LEN * sizeof(float));
    cudaMemset(d_im, 0, HISTORY_LEN * sizeof(float));

    // Events: total spans H2D + kernel + D2H; kernel spans kernel only
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // --- H2D copy ---
    cudaEventRecord(start_total);
    cudaMemcpy(d_re + HISTORY_LEN, h_re.data(), frameLen * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im + HISTORY_LEN, h_im.data(), frameLen * sizeof(float), cudaMemcpyHostToDevice);

    // --- Kernel ---
    const int gridSize = (outputCapacity + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    resamplerComputeStep_Host(THREADS_PER_BLOCK, gridSize, iFs, oFs);

    cudaEventRecord(start_kernel);
    resampler<<<gridSize, THREADS_PER_BLOCK>>>(d_out_re, d_out_im, d_outLen, d_re, d_im, iFs, oFs, frameLen);
    cudaEventRecord(stop_kernel);

    // --- D2H copy ---
    int outLen{};
    cudaMemcpy(&outLen, d_outLen, sizeof(int), cudaMemcpyDeviceToHost);

    PinnedFloatVector h_out_re(outputCapacity), h_out_im(outputCapacity);
    cudaMemcpy(h_out_re.data(), d_out_re, outputCapacity * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_im.data(), d_out_im, outputCapacity * sizeof(float), cudaMemcpyDeviceToHost);
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

    const double rate_kernel = outLen / (ms_kernel * 1e3);
    const double rate_total  = outLen / (ms_total  * 1e3);

    std::cout << "\n--- Timing (" << frameLen << " input -> " << outLen
              << " output samples, iFs=" << iFs << " oFs=" << oFs << ") ---\n"
              << "Kernel only          : " << ms_kernel << " ms"
              << "  |  " << rate_kernel << " MSamples/s\n"
              << "H2D + kernel + D2H   : " << ms_total  << " ms"
              << "  |  " << rate_total  << " MSamples/s\n\n";

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    recordData(h_out_re.data(), h_out_im.data(), outLen, outFile);

    cudaFree(d_re);
    cudaFree(d_im);
    cudaFree(d_out_re);
    cudaFree(d_out_im);
    cudaFree(d_outLen);

    return 0;
}
