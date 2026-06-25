#include <cufft.h>

#define MAX_FILTER_LENGTH 1024

__constant__ float c_baseBandFilter_Coef[MAX_FILTER_LENGTH];

__global__
void filterBaseband(float *d_data_Re, float *d_data_Im, float *d_filteredData_Re,
                    float *d_filteredData_Im, const int filterLength, const int dataLength, float *d_ABS)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum_re{ 0 }, sum_im{ 0 };
    while (i < dataLength)
    {
        sum_re = 0.0f;
        sum_im = 0.0f;

        for (int j = 0; j < filterLength; j++)
        {

            sum_re += c_baseBandFilter_Coef[j] * d_data_Re[i - j + filterLength - 1];
            sum_im += c_baseBandFilter_Coef[j] * d_data_Im[i - j + filterLength - 1];
        }

        d_filteredData_Re[i] = sum_re;
        d_filteredData_Im[i] = sum_im;
        d_ABS[i] = (sum_re*sum_re + sum_im*sum_im);
        i += blockDim.x * gridDim.x;
    }
}


// ---- FFT-based baseband filter (cuFFT) ---------------------------------
// Linear convolution of the IQ signal with the (real) filter coefficients,
// computed as IFFT(FFT(signal) * FFT(filter)) over a single transform
// spanning the whole signal, followed by extraction of the valid filtered
// samples. Selected in bbf_main.cu via USE_FFT_FILTER.

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
        out[i] = (i < filterLength) ? make_cuFloatComplex(c_baseBandFilter_Coef[i], 0.0f)
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

// Extracts the valid linear-convolution region (offset by filterLength-1,
// matching filterBaseband's indexing) and the per-sample magnitude-squared
__global__
void extractFiltered(const cufftComplex *__restrict__ conv,
                     float *__restrict__ outRe, float *__restrict__ outIm, float *__restrict__ outAbs,
                     const int filterLength, const int outputLength)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < outputLength)
    {
        const cufftComplex c = conv[i + filterLength - 1];
        outRe[i]  = c.x;
        outIm[i]  = c.y;
        outAbs[i] = c.x * c.x + c.y * c.y;
    }
}

