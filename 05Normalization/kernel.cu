__device__ __managed__ float d_NormalizeFactor{1},d_NormalizeFactor_MF{1}, d_Mean_FSK{};
__device__ __managed__ int d_nValid_vec[1];

// Per-sample power: pow(abs(sample),2) = Re^2 + Im^2. Feeds both the sum
// reduction (UnitPowerNormalization) and the max reduction (AGC) below.
__global__
void computePowerSamples(const float *__restrict__ d_data_Re, const float *__restrict__ d_data_Im,
                         float *__restrict__ d_power, const bool isOqpsk = false, const int dataLen = 0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int len{ isOqpsk ? dataLen : d_nValid_vec[0] };
    while (i < len)
    {
        d_power[i] = d_data_Re[i] * d_data_Re[i] + d_data_Im[i] * d_data_Im[i];
        i += blockDim.x * gridDim.x;
    }
}

__global__
void NF_LoopFilter_MF(float *d_sum, const bool isOqpsk = false, const int dataLen = 0)
{
    int denum{ isOqpsk ? dataLen : d_nValid_vec[0]};
    d_NormalizeFactor_MF = sqrtf(d_sum[0] / denum);
    if(d_NormalizeFactor_MF == 0)
        d_NormalizeFactor_MF = 1;

    d_sum[0] = 0;
}

// AGC counterpart of NF_LoopFilter_MF: scales so the block's peak amplitude
// (not its average power) maps to 1. d_max[0] is expected to hold the max of
// computePowerSamples' output (squared magnitude), so sqrt gives the peak
// amplitude directly.
__global__
void NF_LoopFilter_AGC(float *d_max, const bool isOqpsk = false, const int dataLen = 0)
{
    d_NormalizeFactor_MF = sqrtf(d_max[0]);
    if(d_NormalizeFactor_MF == 0)
        d_NormalizeFactor_MF = 1;

    d_max[0] = 0;
}

__global__
void Normalize_MF(float *d_data_Re, float *d_data_Im, float *d_normalize_Re, float *d_normalize_Im,
                  const bool isOqpsk = false, const int dataLen = 0)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int nOut{ isOqpsk ? dataLen : d_nValid_vec[0] };
    while (i < nOut)
    {
        d_normalize_Re[i] = d_data_Re[i] / d_NormalizeFactor_MF;
        d_normalize_Im[i] = d_data_Im[i] / d_NormalizeFactor_MF;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void parallelSum_LenDivisibleBy1024(float *d_data, float *d_Sum)
{
    __shared__ float sdata[1024];

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = d_data[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        atomicAdd(d_Sum, sdata[0]);
    }
}

__global__ void parallelSum_arbitraryLen(float *a, float* sum, const bool isOqpsk = false, const int dataLen = 0) {
    int dataLength{ isOqpsk ? dataLen : d_nValid_vec[0] };
    unsigned int i{ threadIdx.x + blockDim.x * blockIdx.x };
    if (i >= dataLength)
        return;

    unsigned int i_thr{ threadIdx.x };
    const int blockLen{ static_cast<int>(blockDim.x) };
    int prevPowOf2{ blockLen };

    if ((blockIdx.x == gridDim.x - 1) && (dataLength % blockLen != 0)) {
        prevPowOf2 = powf(2, floorf(log2f(dataLength % blockLen)));
        if (i_thr < dataLength % blockLen - prevPowOf2)
            a[i] += a[i + prevPowOf2];
        __syncthreads();
    }

    int numActive{ prevPowOf2 / 2 };
    for (int j{}; j < log2f(prevPowOf2); j++) {
        if (i_thr  < numActive)
            a[i] += a[i + numActive];
        numActive /= 2;
        __syncthreads();
    }
    if (i_thr == 0) {
        atomicAdd(sum, a[blockDim.x*blockIdx.x]);
    }
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void parallelMax_arbitraryLen(float *a, float* sum, const bool isOqpsk = false, const int dataLen = 0) {
//    __shared__ float a_shared[1024];
    int dataLength{ isOqpsk ? dataLen : d_nValid_vec[0] };
    unsigned int i{ threadIdx.x + blockDim.x * blockIdx.x };
    if (i >= dataLength)
        return;

    unsigned int i_thr{ threadIdx.x };
//    a_shared[i_thr]=a[i];
    const int blockLen{ static_cast<int>(blockDim.x) };
    int prevPowOf2{ blockLen };

    if ((blockIdx.x == gridDim.x - 1) && (dataLength % blockLen != 0)) {
        prevPowOf2 = powf(2, floorf(log2f(dataLength % blockLen)));
        if (i_thr < dataLength % blockLen - prevPowOf2)
            a[i] = fmax(a[i],a[i + prevPowOf2]);
        __syncthreads();
    }

    int numActive{ prevPowOf2 / 2 };
    for (int j{}; j < log2f(prevPowOf2); j++) {
        if (i_thr  < numActive)
            a[i] = fmax(a[i],a[i + numActive]);
        numActive /= 2;
        __syncthreads();
    }
    if (i_thr == 0) {
        atomicMax(sum, a[blockDim.x*blockIdx.x]);
    }
}
