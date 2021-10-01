#include "kernel.cuh"


//------------------ Cuda Params ------------------//
__constant__ float C_input_sync_data[SyncLength*NumSync];

__constant__ float dev_x[8], dev_y[8], dev_m[8];
__constant__ int dev_bits[16][4];
__constant__ int dev_prev_state[16][2], dev_next_state[16][2];

//------------------ ProtoTypes ------------------//
cudaError_t mainFunc(float *output, DataIn input_data, Params sigParams);


//------------------ Kernels ------------------//

__global__
void interleaver(float *output, const float *input, const int *interleaverMat, const int *dataLen, const int maxLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	int blockLen = dataLen[i_y];
	__shared__ int startInd;
	if (i_y < numBlock)
	{
		if (i_x == 0)
		{
			startInd = 0;
			for (int i = 0; i < i_y; i++)
			{
				startInd += dataLen[i];
			}
			//printf("%i %i \n", i_y, startInd);
		}
		__syncthreads();

		while (i_x < blockLen)
		{
			output[i_x + i_y*maxLen] = input[interleaverMat[i_x + startInd] + i_y*maxLen];
			i_x += blockDim.x;
		}
	}
}


__global__
void deinterleaver_n(float *output, const float *input, const int *interleaverMat, const int *dataLen, const int maxLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	int blockLen = dataLen[i_y];
	__shared__ int startInd;
	if (i_y < numBlock)
	{
		if (i_x == 0)
		{
			startInd = 0;
			for (int i = 0; i < i_y; i++)
			{
				startInd += dataLen[i];
			}
			//printf("%i %i \n", i_y, startInd);
		}
		__syncthreads();
		while (i_x < blockLen)
		{
			output[interleaverMat[i_x + startInd] + i_y*maxLen] = input[i_x + i_y*maxLen];
			i_x += blockDim.x;
		}
	}
}

__global__
void fill_InputLLR(float *LLR1, float *LLR2, const float *sys1, const float *sys2, const float *parity1, const float *parity2, const int dataLen)
{
	int i_x = threadIdx.x;
	while (i_x < dataLen)
	{
		LLR1[2 * i_x + 0] = sys1[i_x];
		LLR1[2 * i_x + 1] = parity1[i_x];
		LLR2[2 * i_x + 0] = sys2[i_x];
		LLR2[2 * i_x + 1] = parity2[i_x];

		i_x += blockDim.x;
	}
}


__global__
void myfunc_n(float *tmp, const float *input_llr, const float *sys, const float *parity, const int *fecLen, const int maxFecLen, const int numBlock)
{

	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	int maxLen{ maxFecLen };

	if (i_y < numBlock)
	{
		while (i_x < fecLen[i_y])
		{
			int index{ 7 };
			float output{ 0 };
			for (int k = 0; k < 8; k++)
				if (input_llr[i_x + i_y*maxLen] <= dev_x[k])
				{
					index = k;
					break;
				}
			output += dev_m[index] * (input_llr[i_x + i_y*maxLen] - dev_x[index]) + dev_y[index];

			index = 7;
			for (int k = 0; k < 8; k++)
				if (sys[i_x + i_y*maxLen] <= dev_x[k])
				{
					index = k;
					break;
				}
			output += dev_m[index] * (sys[i_x + i_y*maxLen] - dev_x[index]) + dev_y[index];


			index = 7;
			for (int k = 0; k < 8; k++)
				if (parity[i_x + i_y*maxLen] <= dev_x[k])
				{
					index = k;
					break;
				}
			output += dev_m[index] * (parity[i_x + i_y*maxLen] - dev_x[index]) + dev_y[index];

			tmp[i_x + i_y*maxLen] = output;
			i_x += blockDim.x;
		}
	}
}

__global__
void set_Gamma_n(float *gamma, float *input_llr, float *sys, float *parity, float *tmp, const int *length, const int n_state, const int maxFecLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	int j = threadIdx.y;
	int maxLen = maxFecLen;
	if (i_y < numBlock)
	{
		while (i_x < length[i_y])
		{
			if (j < n_state)
			{
				gamma[2 * (i_x*n_state + j) + 0 + 2 * i_y*n_state*maxFecLen] = tmp[i_x + i_y*maxFecLen] + 0 + (dev_bits[j][0] == 1 ? sys[i_x + i_y*maxFecLen] : 0) + (dev_bits[j][1] == 1 ? parity[i_x + i_y*maxFecLen] : 0);
				gamma[2 * (i_x*n_state + j) + 1 + 2 * i_y*n_state*maxFecLen] = tmp[i_x + i_y*maxFecLen] + input_llr[i_x + i_y*maxFecLen] + (dev_bits[j][2] == 1 ? sys[i_x + i_y*maxFecLen] : 0) + (dev_bits[j][3] == 1 ? parity[i_x + i_y*maxFecLen] : 0);
			}
			i_x += blockDim.x;
		}
	}
}


__global__
void set_AlphaBeta_n(float *beta, float *alpha, float *gamma, const int *length, const int n_state, const int denum, const int maxFecLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x / 2;
	int idx, idx_, idx_1;
	float v1, v2;
	int jj_;
	if (i_y < numBlock)
	{
		if ((blockIdx.x % 2) == 0 && i_x < n_state)
		{
			if (i_x == 0)
			{
				alpha[0 + n_state*i_y*maxFecLen] = 0;
			}
			else if (i_x < 16)
			{
				alpha[i_x + n_state*i_y*maxFecLen] = -1e5;
			}
			__syncthreads();

			for (int j = 1; j < length[i_y]; j++)
			{
				idx = j * n_state;
				idx_ = (j - 1) * n_state;
				idx_1 = 2 * n_state * (j - 1);

				v1 = alpha[dev_prev_state[i_x][0] + idx_ + n_state*i_y*maxFecLen] + gamma[2 * dev_prev_state[i_x][0] + idx_1 + 2 * n_state*i_y*maxFecLen];
				v2 = alpha[dev_prev_state[i_x][1] + idx_ + n_state*i_y*maxFecLen] + gamma[2 * dev_prev_state[i_x][1] + 1 + idx_1 + 2 * n_state*i_y*maxFecLen];
				alpha[idx + i_x + n_state*i_y*maxFecLen] = v1 > v2 ? v1 : v2;
				__syncthreads();

			}
		}

		if ((blockIdx.x % 2) == 1 && i_x < n_state)
		{
			int j{ length[i_y] - 1 };

			if (i_x == 0)
				beta[j * n_state + n_state*i_y* maxFecLen] = 0;
			else if (i_x < n_state && denum == 1)
				beta[j * n_state + i_x + n_state*i_y* maxFecLen] = -1e5;
			__syncthreads();

			for (j = length[i_y] - 2; j >= 0; j--)
			{
				idx = j * n_state;
				idx_ = (j + 1) * n_state;
				idx_1 = 2 * n_state * (j + 1);
				jj_ = 2 * i_x + idx_1;
				v1 = beta[dev_next_state[i_x][0] + idx_ + n_state*i_y*maxFecLen] + gamma[jj_ + 2 * n_state*i_y* maxFecLen];
				v2 = beta[dev_next_state[i_x][1] + idx_ + n_state*i_y* maxFecLen] + gamma[jj_ + 1 + 2 * n_state*i_y* maxFecLen];
				beta[idx + i_x + n_state*i_y* maxFecLen] = v1 > v2 ? v1 : v2;
				__syncthreads();
			}
		}
	}
}

__global__
void calc_Inp_Out_n(float* output, float *input, const float *sys, const float *gamma, const float *alpha, const float *beta, const int *length, const int n_state, const int maxFecLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;

	if (i_y < numBlock)
	{
		while (i_x < length[i_y])
		{
			int  idx_, idx_1, idx_2, idx_3, idx_4;

			float temp1 = 0;
			float temp2 = 0;

			idx_ = i_x*n_state;
			idx_1 = 2 * n_state*i_x;
			idx_2 = idx_ + 1;

			temp1 = alpha[idx_ + n_state*i_y*maxFecLen] + gamma[idx_1 + 1 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[0][1] + idx_ + n_state*i_y*maxFecLen];
			temp2 = alpha[idx_2 + n_state*i_y*maxFecLen] + gamma[idx_1 + 3 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[1][1] + idx_ + n_state*i_y*maxFecLen];
			float num = temp1 > temp2 ? temp1 : temp2;

			temp1 = alpha[idx_ + n_state*i_y*maxFecLen] + gamma[idx_1 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[0][0] + idx_ + n_state*i_y*maxFecLen];
			temp2 = alpha[idx_2 + n_state*i_y*maxFecLen] + gamma[idx_1 + 2 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[1][0] + idx_ + n_state*i_y*maxFecLen];
			float denum = temp1 > temp2 ? temp1 : temp2;

			for (int jj = 2; jj < n_state; jj++)
			{
				idx_3 = idx_ + jj;
				idx_4 = idx_1 + 2 * jj;

				temp1 = alpha[idx_3 + n_state*i_y*maxFecLen] + gamma[idx_4 + 1 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[jj][1] + idx_ + n_state*i_y*maxFecLen];
				if (num < temp1)
					num = temp1;

				temp1 = alpha[idx_3 + n_state*i_y*maxFecLen] + gamma[idx_4 + 2 * n_state*i_y*maxFecLen] + beta[dev_next_state[jj][0] + idx_ + n_state*i_y*maxFecLen];
				if (denum < temp1)
					denum = temp1;
			}
			output[i_x + i_y*maxFecLen] = num - denum;
			input[i_x + i_y*maxFecLen] = output[i_x + i_y*maxFecLen] - input[i_x + i_y*maxFecLen] - sys[i_x + i_y*maxFecLen];

			i_x += blockDim.x;
		}
	}
}

__global__
void limiter_n(float* output, float *input, const int *len, const int maxLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	if (i_y < numBlock)
	{
		while (i_x < len[i_y])
		{
			if (input[i_x + i_y*maxLen] > 35)
				output[i_x + i_y*maxLen] = 35;
			else if (input[i_x + i_y*maxLen] < -35)
				output[i_x + i_y*maxLen] = -35;
			else
				output[i_x + i_y*maxLen] = input[i_x + i_y*maxLen];
			i_x += blockDim.x;
		}
	}
}

__global__
void hardDecision_n(float *output, const int *dataLen, const int maxLen, const int numBlock)
{
	int i_x = threadIdx.x;
	int i_y = blockIdx.x;
	if (i_y < numBlock)
	{
		while (i_x < dataLen[i_y])
		{
			output[i_x + i_y*maxLen] = (output[i_x + i_y*maxLen] > 0 ? 1 : 0);
			i_x += blockDim.x;
		}
	}
}

cudaError_t mainFunc(float *output, DataIn input_data, Params sigParams)
{
	
		
			// ------------------- Turbo Decoder (Part 2)--------------- //


		
			cudaStatus = cudaMemcpyAsync(dev_fecblockLen + 1, sigParams.FecBlockLen + 1, (sigParams.DeinterLeaverDepth - 1) * sizeof(int), cudaMemcpyHostToDevice);



			for (int i = 0; i < sigParams.DeinterLeaverDepth - 1; i++)
			{

				cudaStatus = cudaMemcpy(dev_sys1_n + i*sigParams.MaxFECLen, dev_eightFecBlockData + (3 * i + 3)*sigParams.MaxFECLen, sigParams.MaxFECLen * sizeof(float), cudaMemcpyDeviceToDevice);
				cudaStatus = cudaMemcpy(dev_parity1_n + i*sigParams.MaxFECLen, dev_eightFecBlockData + (3 * i + 3)*sigParams.MaxFECLen + sigParams.FecBlockLen[i + 1], sigParams.MaxFECLen * sizeof(float), cudaMemcpyDeviceToDevice);
				cudaStatus = cudaMemcpy(dev_parity2_n + i*sigParams.MaxFECLen, dev_eightFecBlockData + (3 * i + 3) * sigParams.MaxFECLen + 2 * sigParams.FecBlockLen[i + 1], sigParams.MaxFECLen * sizeof(float), cudaMemcpyDeviceToDevice);
			}

			interleaver << <7, threadsPerBlock_interLeaveTurbo >> > (dev_sys2_n, dev_sys1_n, dev_inter_turbo_pattern + sigParams.FecBlockLen[0], dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Sys2 2-8 FEC Turbo failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}



			int DecoderIterations2 = 4;
			for (int i_turboDecode = 0; i_turboDecode < DecoderIterations2; i_turboDecode++)
			{

				myfunc_n << <7, threadsPerBlock_turboCalc >> > (dev_tmp_n, dev_intrinsic_LLR_n, dev_sys1_n, dev_parity1_n, dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				set_Gamma_n << < 7, threadsPerBlock_turbo_gamma >> > (dev_gamma_n, dev_intrinsic_LLR_n, dev_sys1_n, dev_parity1_n, dev_tmp_n, dev_fecblockLen + 1, 16, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				set_AlphaBeta_n << < 7 * 2, 16 >> > (dev_beta_n, dev_alpha_n, dev_gamma_n, dev_fecblockLen + 1, 16, 2, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				calc_Inp_Out_n << < 7, threadsPerBlock_turboCalc >> > (dev_output_LLR_n, dev_intrinsic_LLR_n, dev_sys1_n, dev_gamma_n, dev_alpha_n, dev_beta_n, dev_fecblockLen + 1, 16, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);

				limiter_n << < 7, threadsPerBlock_turboCalc >> > (dev_output_LLR_n, dev_intrinsic_LLR_n, dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				interleaver << <7, threadsPerBlock_turboCalc >> > (dev_intrinsic_LLR_n, dev_output_LLR_n, dev_inter_turbo_pattern + sigParams.FecBlockLen[0], dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);

				myfunc_n << <7, threadsPerBlock_turboCalc >> > (dev_tmp_n, dev_intrinsic_LLR_n, dev_sys2_n, dev_parity2_n, dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				set_Gamma_n << < 7, threadsPerBlock_turbo_gamma >> > (dev_gamma_n, dev_intrinsic_LLR_n, dev_sys2_n, dev_parity2_n, dev_tmp_n, dev_fecblockLen + 1, 16, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				set_AlphaBeta_n << < 7 * 2, 16 >> > (dev_beta_n, dev_alpha_n, dev_gamma_n, dev_fecblockLen + 1, 16, 2, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				calc_Inp_Out_n << < 7, threadsPerBlock_turboCalc >> > (dev_output_LLR_n, dev_intrinsic_LLR_n, dev_sys2_n, dev_gamma_n, dev_alpha_n, dev_beta_n, dev_fecblockLen + 1, 16, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);

				if (i_turboDecode == DecoderIterations - 1)
				{
					limiter_n << < 7, threadsPerBlock_turboCalc >> > (dev_intrinsic_LLR_n, dev_output_LLR_n, dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
					deinterleaver_n << <7, threadsPerBlock_turboCalc >> > (dev_output_LLR_n, dev_intrinsic_LLR_n, dev_inter_turbo_pattern + sigParams.FecBlockLen[0], dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				}
				else
				{
					limiter_n << < 7, threadsPerBlock_turboCalc >> > (dev_output_LLR_n, dev_intrinsic_LLR_n, dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
					deinterleaver_n << <7, threadsPerBlock_turboCalc >> > (dev_intrinsic_LLR_n, dev_output_LLR_n, dev_inter_turbo_pattern + sigParams.FecBlockLen[0], dev_fecblockLen + 1, sigParams.MaxFECLen, sigParams.DeinterLeaverDepth - 1);
				}
		
			}

			cudaStatus = cudaGetLastError();
			

		// ------------------- End --------------- //


	}



	
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_out);
	cudaFree(dev_data_re);
	cudaFree(dev_data_im);

	return cudaStatus;
}

