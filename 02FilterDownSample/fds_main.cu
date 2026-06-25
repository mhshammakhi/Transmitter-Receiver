#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

#include "../utils/utils.h"

#define MAX_FILTER_LENGTH 1024

// Decimation-filter implementation select:
//   0 = direct time-domain FIR convolution, computed only at the decimated
//       output positions (default)
//   1 = FFT-based convolution using cuFFT: a single transform spanning the
//       whole signal, frequency-domain multiply, inverse transform, decimate
#define USE_FFT_FILTER 0

// Defined in kernel.cu
extern __constant__ float c_decimationFilter_Coef[];

__global__
void filterAndDownSample(float *d_data_Re, float *d_data_Im,
                         float *d_filteredData_Re, float *d_filteredData_Im,
                         const int filterLength, const int dataLength,
                         const int decimationFactor);

__global__
void packSignalComplex(const float *re, const float *im, cufftComplex *out,
                       const int len, const int fftSize);
__global__
void packFilterComplex(cufftComplex *out, const int filterLength, const int fftSize);
__global__
void complexMultiplyScale(cufftComplex *a, const cufftComplex *b, const int fftSize);
__global__
void extractDownsample(const cufftComplex *conv, float *outRe, float *outIm,
                       const int filterLength, const int outputLength,
                       const int decimationFactor);

static constexpr int THREADS_PER_BLOCK = 256;

#if USE_FFT_FILTER
static int nextPow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}
#endif

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <output.bin> <coeff.bin> <decimationFactor>\n"
                  << "  coeff.bin       : binary file of float32 filter coefficients\n"
                  << "  decimationFactor: integer downsampling ratio (e.g. 4)\n";
        return 1;
    }

    const std::string inFile           = argv[1];
    const std::string outFile          = argv[2];
    const std::string coeffFile        = argv[3];
    const int         decimationFactor = std::stoi(argv[4]);

    // Load filter coefficients from binary file
    std::ifstream cf(coeffFile, std::ios::binary);
    if (!cf) { std::cerr << "Cannot open coeff file: " << coeffFile << "\n"; return 1; }
    cf.seekg(0, std::ios::end);
    const int filterLength = static_cast<int>(cf.tellg() / sizeof(float));
    cf.seekg(0, std::ios::beg);

    if (filterLength > MAX_FILTER_LENGTH)
    {
        std::cerr << "Filter length " << filterLength
                  << " exceeds MAX_FILTER_LENGTH=" << MAX_FILTER_LENGTH << "\n";
        return 1;
    }

    std::vector<float> h_coeff(filterLength);
    cf.read(reinterpret_cast<char *>(h_coeff.data()), filterLength * sizeof(float));
    cf.close();

    cudaMemcpyToSymbol(c_decimationFilter_Coef, h_coeff.data(), filterLength * sizeof(float));

    // Load interleaved IQ binary data into pinned host memory
    PinnedFloatVector h_re, h_im;
    readBinData(h_re, h_im, inFile);
    const int inputLength = static_cast<int>(h_re.size());

    // Output sample count: each output sample i maps to input range centred at i*D
    const int outputLength = (inputLength - filterLength) / decimationFactor + 1;
    if (outputLength <= 0)
    {
        std::cerr << "Input too short for filterLength=" << filterLength
                  << " and decimationFactor=" << decimationFactor << "\n";
        return 1;
    }

    // Allocate device buffers
    float *d_re{}, *d_im{}, *d_out_re{}, *d_out_im{};
    cudaMalloc(&d_re,     inputLength  * sizeof(float));
    cudaMalloc(&d_im,     inputLength  * sizeof(float));
    cudaMalloc(&d_out_re, outputLength * sizeof(float));
    cudaMalloc(&d_out_im, outputLength * sizeof(float));

    // Events: total spans H2D + kernel + D2H; kernel spans kernel only
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // --- H2D copy ---
    cudaEventRecord(start_total);
    cudaMemcpy(d_re, h_re.data(), inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im, h_im.data(), inputLength * sizeof(float), cudaMemcpyHostToDevice);

    // --- Kernel ---
#if USE_FFT_FILTER
    const int fftSize = nextPow2(inputLength + filterLength - 1);
    cufftComplex *d_sigFFT{}, *d_filtFFT{};
    cudaMalloc(&d_sigFFT,  fftSize * sizeof(cufftComplex));
    cudaMalloc(&d_filtFFT, fftSize * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan1d(&plan, fftSize, CUFFT_C2C, 1);

    const int gridSizeFFT = (fftSize     + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int gridSizeOut = (outputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start_kernel);
    packSignalComplex<<<gridSizeFFT, THREADS_PER_BLOCK>>>(d_re, d_im, d_sigFFT, inputLength, fftSize);
    packFilterComplex<<<gridSizeFFT, THREADS_PER_BLOCK>>>(d_filtFFT, filterLength, fftSize);

    cufftExecC2C(plan, d_sigFFT,  d_sigFFT,  CUFFT_FORWARD);
    cufftExecC2C(plan, d_filtFFT, d_filtFFT, CUFFT_FORWARD);

    complexMultiplyScale<<<gridSizeFFT, THREADS_PER_BLOCK>>>(d_sigFFT, d_filtFFT, fftSize);

    cufftExecC2C(plan, d_sigFFT, d_sigFFT, CUFFT_INVERSE);

    extractDownsample<<<gridSizeOut, THREADS_PER_BLOCK>>>(d_sigFFT, d_out_re, d_out_im,
                                                           filterLength, outputLength,
                                                           decimationFactor);
    cudaEventRecord(stop_kernel);
#else
    const int gridSize = (outputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(start_kernel);
    filterAndDownSample<<<gridSize, THREADS_PER_BLOCK>>>(d_re, d_im, d_out_re, d_out_im,
                                                          filterLength, outputLength,
                                                          decimationFactor);
    cudaEventRecord(stop_kernel);
#endif

    // --- D2H copy ---
    PinnedFloatVector h_out_re(outputLength), h_out_im(outputLength);
    cudaMemcpy(h_out_re.data(), d_out_re, outputLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_im.data(), d_out_im, outputLength * sizeof(float), cudaMemcpyDeviceToHost);
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

    const double rate_kernel = outputLength / (ms_kernel * 1e3);
    const double rate_total  = outputLength / (ms_total  * 1e3);

    std::cout << "\n--- Timing (" << inputLength << " input -> " << outputLength
              << " output samples, D=" << decimationFactor << ") ---\n"
              << "Kernel only          : " << ms_kernel << " ms"
              << "  |  " << rate_kernel << " MSamples/s\n"
              << "H2D + kernel + D2H   : " << ms_total  << " ms"
              << "  |  " << rate_total  << " MSamples/s\n\n";

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    recordData(h_out_re.data(), h_out_im.data(), outputLength, outFile);

#if USE_FFT_FILTER
    cufftDestroy(plan);
    cudaFree(d_sigFFT);
    cudaFree(d_filtFFT);
#endif

    cudaFree(d_re);
    cudaFree(d_im);
    cudaFree(d_out_re);
    cudaFree(d_out_im);

    return 0;
}
