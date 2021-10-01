//============================= LDPC ============================//
__global__ void initVariableNode(sparseMatrix * const H, entry* mem, LLR_Type* const l, const int *H_ind, int *n_remained,
                                 int *c, int *iter_arr, const int n_frames)
{
    int thrd_idx = threadIdx.x + (blockIdx.x % blocksPerFrame) * blockDim.x;
    int block_idx = blockIdx.x/ blocksPerFrame;

    int h_ind = H_ind[block_idx];

    if (thrd_idx == 0 && block_idx == 0)
        n_remained[0] = n_frames;

    if (thrd_idx == 0) {
        c[block_idx] = 1;
        iter_arr[block_idx] = 1;
    }

    while (thrd_idx < (H + h_ind)->n_cols) {
        for (entry *e = sm_first_in_col1(mem + memLen * block_idx, H + h_ind, thrd_idx);
             !sm_at_end1(e);
             e = sm_next_in_col1(mem + memLen * block_idx, e))
        {
            e->q = l[thrd_idx + nMax*block_idx];
        }
        thrd_idx += blockDim.x * blocksPerFrame;
    }
}

__global__ void iterCheckNode(sparseMatrix * const H, entry* mem, const int *H_ind, int *c)
{
    int idx = threadIdx.x+ (blockIdx.x % blocksPerFrame) * blockDim.x;
    int block_idx = blockIdx.x/ blocksPerFrame;
    if (c[block_idx] == 0)
        return;

    int parity;
    //depends on the initial values if these two what the fuck
    LLR_Type min1, min2, temp;

    int h_ind = H_ind[block_idx];

    while (idx < (H + h_ind)->n_rows) {
        parity = 1;
        temp = 0;
        min1 = min2 = 32767;
        for (entry * e = sm_first_in_row1(mem + memLen * block_idx, H + h_ind, idx);
             !sm_at_end1(e);
             e = sm_next_in_row1(mem + memLen * block_idx, e))
        {
            parity *= ((e->q >= 0) - (e->q < 0));
            temp = abs(e->q) - betta;
            if (temp < min1) {
                min2 = min1;
                min1 = temp;
            }
            else if ((temp < min2) && (temp > min1)) {
                min2 = temp;
            }
        }

        if (min1 > 1000)
            min1 = 1000;
        if (min1 < -1000)
            min1 = -1000;
        if (min2 > 1000)
            min2 = 1000;
        if (min2 < -1000)
            min2 = -1000;

        for (entry * e = sm_first_in_row1(mem + memLen * block_idx, H + h_ind, idx);
             !sm_at_end1(e);
             e = sm_next_in_row1(mem + memLen * block_idx, e))
        {
            if ((abs(e->q) - betta) == min1) {
                if (min2 >= 0)
                    e->q = ((e->q >= 0) - (e->q < 0)) * parity * min2;
                else
                    e->q = 0;
            }
            else {
                if (min1 >= 0)
                    e->q = ((e->q >= 0) - (e->q < 0)) * parity * min1;
                else
                    e->q = 0;
            }

            /*if (isinf(e->q) || (fabsf(e->q) == 100000000))
            {
            if (e->q < 0)
            e->q = -1000;
            else
            e->q = 1000;
            }*/
        }
        idx += blockDim.x * blocksPerFrame;
    }
}

__global__ void iterVariableNode(sparseMatrix * const H, entry *mem, int *H_ind, LLR_Type *l, char* codeword, int *c)
{
    int idx = threadIdx.x+ (blockIdx.x % blocksPerFrame) * blockDim.x;
    int b_idx = blockIdx.x / blocksPerFrame;
    if (c[b_idx] == 0)
        return;

    int16_t sum;
    int h_ind = H_ind[b_idx];

    while (idx < (H + h_ind)->n_cols) {
        sum = l[idx + nMax * b_idx];
        for (entry *e = sm_first_in_col1(mem + memLen * b_idx, H + h_ind, idx);
             !sm_at_end1(e);
             e = sm_next_in_col1(mem + memLen * b_idx, e))
        {
            sum += e->q;
        }
        if (sum >= 0)
            codeword[idx + nMax * b_idx] = '0';
        else
            codeword[idx + nMax * b_idx] = '1';

        for (entry *e = sm_first_in_col1(mem + memLen * b_idx, H + h_ind, idx);
             !sm_at_end1(e);
             e = sm_next_in_col1(mem + memLen * b_idx, e))
        {
            e->q = sum - e->q;
        }
        idx += blockDim.x * blocksPerFrame;
    }
}

__global__ void check(sparseMatrix * const H, int *H_ind, char* const codeword, int* const c)
{
    int idx = threadIdx.x + (blockIdx.x % blocksPerFrame) * blockDim.x;
    int b_idx = blockIdx.x / blocksPerFrame;
    if (c[b_idx] == 0)
        return;

    if (idx == 0)
        c[b_idx] = 0;

    int sum{};

    int h_ind = H_ind[b_idx];

    while (idx < (H + h_ind)->n_rows) {
        for (entry * e = sm_first_in_row(H + h_ind, idx);
             !sm_at_end(e);
             e = sm_next_in_row(H + h_ind, e))
        {
            sum += codeword[e->col + nMax*b_idx] - '0';
        }
        idx += blockDim.x * blocksPerFrame;
    }
    atomicAdd(c + b_idx, sum % 2);
}

__global__ void check2(int *c, int *iter_arr, int *n_remained, const int n_frames) {
    int idx = threadIdx.x;
    if (idx == 0)
        n_remained[0] = n_frames;
    __syncthreads();

    if (c[idx] != 0)
        iter_arr[idx]++;
    else
        atomicSub(n_remained, 1);
}

__global__ void prepareIntLLR(float *in, LLR_Type *out, const int *len) {
    int idx = threadIdx.x + (blockIdx.x % blocksPerFrame) * blockDim.x;
    int b_idx = blockIdx.x / blocksPerFrame;

    int length = len[b_idx];

    while (idx < length) {
        out[b_idx*nMax + idx] = round(in[b_idx * nMax + idx] * 3);
        idx += blockDim.x * blocksPerFrame;
    }
}
