#include <math_constants.h>

__global__ void DDC(const float *__restrict__ d_data_Re, const float *__restrict__ d_data_Im,
                    float *__restrict__ d_DDC_Re, float *__restrict__ d_DDC_Im,
                    const int dataLength, const float frequency, const float freq_init)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dataLength)
    {
        float omega = fmodf(float(i % 10000) * frequency, 1.0f) * 2.0f * CUDART_PI_F + freq_init;
        float s, c;
        sincosf(omega, &s, &c);
        d_DDC_Re[i] = d_data_Re[i] * c + d_data_Im[i] * s;
        d_DDC_Im[i] = d_data_Im[i] * c - d_data_Re[i] * s;
    }
}
