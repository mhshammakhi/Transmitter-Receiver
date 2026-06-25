#include <cufft.h>

#define MAX_FILTER_LENGTH 1024

__constant__ float c_decimationFilter_Coef[MAX_FILTER_LENGTH];

__global__
void filterAndDownSample(float *d_data_Re, float *d_data_Im,
                         float *d_filteredData_Re,float *d_filteredData_Im,
                         const int filterLength, const int dataLength,
                         const int decimationFactor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum_Re;
    float sum_Im;
    while (i < dataLength)
    {
        sum_Re = 0.0;
        sum_Im = 0.0;

        for (int j = 0; j < filterLength; j++)
        {
            sum_Re += c_decimationFilter_Coef[j] * d_data_Re[i*decimationFactor - j+filterLength-1];
            sum_Im += c_decimationFilter_Coef[j] * d_data_Im[i*decimationFactor - j+filterLength-1];
        }

        d_filteredData_Re[i] = sum_Re;
        d_filteredData_Im[i] = sum_Im;

        i+= blockDim.x * gridDim.x;
    }
}

// ---- FFT-based decimation filter (cuFFT) -------------------------------
// Linear convolution of the IQ signal with the (real) filter coefficients,
// computed as IFFT(FFT(signal) * FFT(filter)) over a single transform
// spanning the whole signal, followed by extraction of the decimated
// output samples. Selected in fds_main.cu via USE_FFT_FILTER.

// Zero-pads the interleaved Re/Im input into a complex buffer of length fftSize
__global__
void packSignalComplex(const float *__restrict__ re, const float *__restrict__ im,
                       cufftComplex *__restrict__ out,
                       const int len, const int fftSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < fftSize)
        out[i] = (i < len) ? make_cuFloatComplex(re[i], im[i])
                            : make_cuFloatComplex(0.0f, 0.0f);
}

// Zero-pads the (real) filter coefficients held in constant memory into a
// complex buffer of length fftSize
__global__
void packFilterComplex(cufftComplex *__restrict__ out,
                       const int filterLength, const int fftSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < fftSize)
        out[i] = (i < filterLength) ? make_cuFloatComplex(c_decimationFilter_Coef[i], 0.0f)
                                     : make_cuFloatComplex(0.0f, 0.0f);
}

// Pointwise frequency-domain multiply (a *= b), with the 1/fftSize scaling
// that cuFFT's unnormalized transforms require folded in
__global__
void complexMultiplyScale(cufftComplex *__restrict__ a, const cufftComplex *__restrict__ b,
                          const int fftSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < fftSize)
    {
        const float scale = 1.0f / fftSize;
        const cufftComplex prod = cuCmulf(a[i], b[i]);
        a[i] = make_cuFloatComplex(prod.x * scale, prod.y * scale);
    }
}

// Extracts every decimationFactor-th sample of the valid linear-convolution
// region (offset by filterLength-1, matching filterAndDownSample's indexing)
__global__
void extractDownsample(const cufftComplex *__restrict__ conv,
                       float *__restrict__ outRe, float *__restrict__ outIm,
                       const int filterLength, const int outputLength,
                       const int decimationFactor)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < outputLength)
    {
        const cufftComplex c = conv[i * decimationFactor + filterLength - 1];
        outRe[i] = c.x;
        outIm[i] = c.y;
    }
}