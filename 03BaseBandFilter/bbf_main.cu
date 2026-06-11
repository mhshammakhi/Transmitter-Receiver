#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "../utils/utils.h"

#define MAX_FILTER_LENGTH 1024

// Defined in kernel.cu
extern __constant__ float c_baseBandFilter_Coef[];

__global__
void filterBaseband(float *d_data_Re, float *d_data_Im,
                    float *d_filteredData_Re, float *d_filteredData_Im,
                    const int filterLength, const int dataLength, float *d_ABS);

static constexpr int THREADS_PER_BLOCK = 256;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <output.bin> <coeff.bin> [abs_out.bin]\n"
                  << "  coeff.bin   : binary file of float32 filter coefficients\n"
                  << "  abs_out.bin : optional; writes magnitude-squared per sample (float32)\n";
        return 1;
    }

    const std::string inFile    = argv[1];
    const std::string outFile   = argv[2];
    const std::string coeffFile = argv[3];
    const std::string absFile   = (argc >= 5) ? argv[4] : "";

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

    cudaMemcpyToSymbol(c_baseBandFilter_Coef, h_coeff.data(), filterLength * sizeof(float));

    // Load interleaved IQ binary data into pinned host memory
    PinnedFloatVector h_re, h_im;
    readBinData(h_re, h_im, inFile);
    const int inputLength = static_cast<int>(h_re.size());

    // Output sample count: kernel accesses d_data[i .. i+filterLength-1], so
    // valid output indices are 0 .. inputLength-filterLength
    const int outputLength = inputLength - filterLength + 1;
    if (outputLength <= 0)
    {
        std::cerr << "Input too short for filterLength=" << filterLength << "\n";
        return 1;
    }

    // Allocate device buffers
    float *d_re{}, *d_im{}, *d_out_re{}, *d_out_im{}, *d_abs{};
    cudaMalloc(&d_re,     inputLength  * sizeof(float));
    cudaMalloc(&d_im,     inputLength  * sizeof(float));
    cudaMalloc(&d_out_re, outputLength * sizeof(float));
    cudaMalloc(&d_out_im, outputLength * sizeof(float));
    cudaMalloc(&d_abs,    outputLength * sizeof(float));

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
    const int gridSize = (outputLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEventRecord(start_kernel);
    filterBaseband<<<gridSize, THREADS_PER_BLOCK>>>(d_re, d_im, d_out_re, d_out_im,
                                                     filterLength, outputLength, d_abs);
    cudaEventRecord(stop_kernel);

    // --- D2H copy ---
    PinnedFloatVector h_out_re(outputLength), h_out_im(outputLength);
    std::vector<float> h_abs(outputLength);
    cudaMemcpy(h_out_re.data(), d_out_re, outputLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_im.data(), d_out_im, outputLength * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_abs.data(),    d_abs,    outputLength * sizeof(float), cudaMemcpyDeviceToHost);
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
              << " output samples) ---\n"
              << "Kernel only          : " << ms_kernel << " ms"
              << "  |  " << rate_kernel << " MSamples/s\n"
              << "H2D + kernel + D2H   : " << ms_total  << " ms"
              << "  |  " << rate_total  << " MSamples/s\n\n";

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);

    recordData(h_out_re.data(), h_out_im.data(), outputLength, outFile);

    if (!absFile.empty())
    {
        std::ofstream af(absFile, std::ios::binary);
        if (af)
        {
            af.write(reinterpret_cast<const char *>(h_abs.data()),
                     outputLength * sizeof(float));
            std::cout << "Magnitude-squared written to " << absFile << "\n";
        }
        else
        {
            std::cerr << "Cannot open abs output file: " << absFile << "\n";
        }
    }

    cudaFree(d_re);
    cudaFree(d_im);
    cudaFree(d_out_re);
    cudaFree(d_out_im);
    cudaFree(d_abs);

    return 0;
}
