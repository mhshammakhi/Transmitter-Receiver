__device__ __managed__ int d_locStep_int{};
__device__ __managed__ float d_locStep_float{};
__device__ __managed__ int d_resample_nFromPrev{};
__device__ __managed__ float d_last_startInterpIndex{ 1 };
__device__ __managed__ float d_resample_prev_re[4]{}, d_resample_prev_im[4]{};

void resamplerComputeStep_Host(const int& num_threads, const int& num_blocks, const float& iFs, const float& oFs) {
    double decimationRate{ (double)iFs / (double)oFs };
    double location_step = num_threads * num_blocks * decimationRate;
    d_locStep_int = floor(location_step);
    d_locStep_float = (float)(location_step - floor(location_step));
}

__device__ __forceinline__ void cubicInterpolate_dev(const float *in_re, const float *in_im, const float& mu, float *out_re, float *out_im) {
    float v_re[4], v_im[4];
    v_re[0] = in_re[2];
    v_re[1] = -1.f / 6 * in_re[0] + in_re[1] - 1.f / 2 * in_re[2] - 1.f / 3 * in_re[3];
    v_re[2] = 1.f / 2 * in_re[1] - in_re[2] + 1.f / 2 * in_re[3];
    v_re[3] = 1.f / 6 * in_re[0] - 1.f / 2 * in_re[1] + 1.f / 2 * in_re[2] - 1.f / 6 * in_re[3];

    v_im[0] = in_im[2];
    v_im[1] = -1.f / 6 * in_im[0] + in_im[1] - 1.f / 2 * in_im[2] - 1.f / 3 * in_im[3];
    v_im[2] = 1.f / 2 * in_im[1] - in_im[2] + 1.f / 2 * in_im[3];
    v_im[3] = 1.f / 6 * in_im[0] - 1.f / 2 * in_im[1] + 1.f / 2 * in_im[2] - 1.f / 6 * in_im[3];

    out_re[0] = ((v_re[3] * mu + v_re[2]) * mu + v_re[1]) * mu + v_re[0];
    out_im[0] = ((v_im[3] * mu + v_im[2]) * mu + v_im[1]) * mu + v_im[0];
}

__global__ void resampler(float *out_re, float *out_im, int *outLen, float *in_re, float *in_im,
                                const float iFs, const float oFs, const int frameLen) {
    float decimationRate{ iFs / oFs };
    int lastIndex{ frameLen + d_resample_nFromPrev - 1 }, startInterpInputIndex{ 1 - d_resample_nFromPrev };
    float startInterpIndex{ d_last_startInterpIndex };
    float temp = (floorf((lastIndex - startInterpIndex) / decimationRate)) * decimationRate + startInterpIndex;
    int outlen{ static_cast<int>(floorf((lastIndex - startInterpIndex) / decimationRate)) + 1 * (temp < (lastIndex - 1)) };

    unsigned int i{ threadIdx.x + blockIdx.x * blockDim.x };
    float location_float{ i * decimationRate - floorf(i * decimationRate) }, locStep_float{ d_locStep_float };
    int location_int{ static_cast<int>(i * decimationRate) }, locStep_int{ d_locStep_int };

    int resample_nFromPrev_local{ d_resample_nFromPrev };
    float interpClockPhase{};
    int interpInputIndex{};
    int lastThrIdx{outlen - int(floor((float)outlen / (blockDim.x * gridDim.x))) * static_cast<int>(blockDim.x * gridDim.x) - 1};

    //if blockDim.x < 4 take this to the while loop
    if (i < resample_nFromPrev_local) {
        in_re[4 - resample_nFromPrev_local + i] = d_resample_prev_re[i];
        in_im[4 - resample_nFromPrev_local + i] = d_resample_prev_im[i];
    }

    while (i < outlen) {
        interpClockPhase = location_float + startInterpIndex - floorf(startInterpIndex);
        interpInputIndex = floorf(interpClockPhase);
        interpClockPhase = 1 - (interpClockPhase - floorf(interpClockPhase));

        interpInputIndex += -1 + location_int + floorf(startInterpIndex) + 4 - resample_nFromPrev_local;
        cubicInterpolate_dev(in_re + interpInputIndex, in_im + interpInputIndex, interpClockPhase, out_re + i, out_im + i);

        location_int += locStep_int;
        location_float += locStep_float;
        i += blockDim.x * gridDim.x;
    }

    if ((threadIdx.x + blockIdx.x * blockDim.x) == lastThrIdx) {
        location_int -= locStep_int;
        location_float -= locStep_float;
        i -= blockDim.x * gridDim.x;
        float temp1{ startInterpIndex + location_int + location_float + decimationRate - lastIndex };
        startInterpInputIndex = static_cast<int>(floorf(temp1)) - 1;
        d_last_startInterpIndex = temp1 + 1 - floorf(temp1);
        d_resample_nFromPrev = 1 - startInterpInputIndex;

        for (int tmp{}; tmp < d_resample_nFromPrev; tmp++) {
            d_resample_prev_re[tmp] = in_re[frameLen + 4 - (d_resample_nFromPrev - tmp)];
            d_resample_prev_im[tmp] = in_im[frameLen + 4 - (d_resample_nFromPrev - tmp)];
        }
        outLen[0] = outlen;
    }
}