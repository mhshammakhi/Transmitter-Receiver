
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "gputimer.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda.h>

#include <time.h>

void dec_turboWithCuda(int *interleaver, float *Sys1, float *Sys2, float *Parity1, float *Parity2, const int decnum, const int length);

__constant__ float dev_x[8], dev_y[8], dev_m[8];
__constant__ int dev_bits[16][4];
__constant__ int dev_prev_state[16][2], dev_next_state[16][2];
__constant__ int dev_interleaver[5124];
__constant__ int block_len[210];


__global__
void initialize_alpha_beta(float *init_alpha, float *init_beta, float *init_alpha_2, float *init_beta_2, const int len)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < len * 16)
	{
		if (i % 16 == 0)
		{
			init_alpha[i] = 0;
			init_alpha_2[i] = 0;
			init_beta[i] = 0;
			init_beta_2[i] = 0;
		}
		else
		{
			init_alpha[i] = -100000;
			init_alpha_2[i] = -100000;
			init_beta[i] = -100000;
			init_beta_2[i] = -100000;
		}
	}
}
__global__
void myfunc(float *tmp, float *input_llr, float *sys, float *parity, const int length)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < length)
	{
		int index{ 7 };
		float output{ 0 };
		for (int k = 0; k < 8; k++)
			if (input_llr[i] <= dev_x[k])
			{
				index = k;
				break;
			}
		output += dev_m[index] * (input_llr[i] - dev_x[index]) + dev_y[index];
		index = 7;
		for (int k = 0; k < 8; k++)
			if (sys[i] <= dev_x[k])
			{
				index = k;
				break;
			}
		output += dev_m[index] * (sys[i] - dev_x[index]) + dev_y[index];


		index = 7;
		for (int k = 0; k < 8; k++)
			if (parity[i] <= dev_x[k])
			{
				index = k;
				break;
			}
		output += dev_m[index] * (parity[i] - dev_x[index]) + dev_y[index];

		tmp[i] = output;
	}
}

__global__
void set_Gamma(float *gamma, float *input_llr, float *sys, float *parity, float *tmp, const int length, const int n_state)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < n_state && j < length)
	{
		gamma[2 * (j*n_state + i)] = tmp[j] + (dev_bits[i][0] == 1 ? sys[j] : 0) + (dev_bits[i][1] == 1 ? parity[j] : 0);
		gamma[2 * (j*n_state + i) + 1] = tmp[j] + input_llr[j] + (dev_bits[i][2] == 1 ? sys[j] : 0) + (dev_bits[i][3] == 1 ? parity[j] : 0);
	}
	/*if (j == 5123)
		printf("i:  thread index: %i %i %f %f\n", i, j, gamma[2 * (j*n_state + i)],gamma[2 * (j*n_state + i) + 1]);*/
		//i += blockDim.x * gridDim.x;
		//j += blockDim.y * gridDim.y;

}

__global__
void set_AlphaBeta(float *alpha_out, float *beta_out, float *gamma, float *init_alpha, float *init_beta, const int length, const int n_block, const int n_state, const int guard_len)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int y = blockIdx.y;

	int len_this_block{ 0 };
	int sub_block_len{ 240 };
	if (j + 1 == n_block)
		len_this_block = sub_block_len + length%sub_block_len;
	else
		len_this_block = sub_block_len + guard_len;
	if (y == 0)
	{
		__shared__ float sh_alpha[10][16];
		for (int k = 0; k < len_this_block; k++)
		{
			if (k == 0 && i < n_state)
			{
				if (j == 0)
				{
					sh_alpha[0][i] = init_alpha[i];
					alpha_out[i] = sh_alpha[0][i];
				}
				else
				{
					int idx = j*sub_block_len * n_state;
					int idx_ = (j*sub_block_len - 1) * n_state;
					int idx_1 = 2 * n_state * (j*sub_block_len - 1);
					float v1 = init_alpha[dev_prev_state[i][0] + idx_] + gamma[2 * dev_prev_state[i][0] + idx_1];
					float v2 = init_alpha[dev_prev_state[i][1] + idx_] + gamma[2 * dev_prev_state[i][1] + 1 + idx_1];
					sh_alpha[0][i] = v1 > v2 ? v1 : v2;
				}
			}
			else if (k < guard_len && i < n_state)
			{
				int idx = (k + j*sub_block_len) * n_state;
				int idx_ = (k - 1 + j*sub_block_len) * n_state;
				int idx_1 = 2 * n_state * (k - 1 + j*sub_block_len);

				float v1 = sh_alpha[k - 1][dev_prev_state[i][0]] + gamma[2 * dev_prev_state[i][0] + idx_1];
				float v2 = sh_alpha[k - 1][dev_prev_state[i][1]] + gamma[2 * dev_prev_state[i][1] + 1 + idx_1];
				sh_alpha[k][i] = v1 > v2 ? v1 : v2;
				if (j == 0)
					alpha_out[idx + i] = sh_alpha[k][i];
			}
			else if (k == guard_len && i < n_state)
			{
				int idx = (k + j*sub_block_len) * n_state;
				int idx_ = (k - 1 + j*sub_block_len) * n_state;
				int idx_1 = 2 * n_state * (k - 1 + j*sub_block_len);

				float v1 = sh_alpha[guard_len - 1][dev_prev_state[i][0]] + gamma[2 * dev_prev_state[i][0] + idx_1];
				float v2 = sh_alpha[guard_len - 1][dev_prev_state[i][1]] + gamma[2 * dev_prev_state[i][1] + 1 + idx_1];
				alpha_out[idx + i] = v1 > v2 ? v1 : v2;
			}
			else if (i < n_state)
			{
				int idx = (k + j*sub_block_len) * n_state;
				int idx_ = (k - 1 + j*sub_block_len) * n_state;
				int idx_1 = 2 * n_state * (k - 1 + j*sub_block_len);

				float v1 = alpha_out[dev_prev_state[i][0] + idx_] + gamma[2 * dev_prev_state[i][0] + idx_1];
				float v2 = alpha_out[dev_prev_state[i][1] + idx_] + gamma[2 * dev_prev_state[i][1] + 1 + idx_1];
				alpha_out[idx + i] = v1 > v2 ? v1 : v2;
			}
			__syncthreads();

		}
	}
	else if (y == 1)
	{
		__shared__ float sh_beta[10][16];
		for (int k = len_this_block - 1; k >= 0; k--)
		{
			if (k == len_this_block - 1 && i < n_state)
			{
				if (j + 1 == n_block)
				{
					sh_beta[0][i] = init_beta[(j*sub_block_len + k)*n_state + i];
					beta_out[(k + j*sub_block_len)* n_state + i] = sh_beta[0][i];
				}
				else
				{
					int idx = (k + j*sub_block_len) * n_state;
					int idx_ = (k + 1 + j*sub_block_len) * n_state;
					int idx_1 = 2 * n_state * (k + 1 + j*sub_block_len);
					float v1 = init_beta[dev_next_state[i][0] + idx_] + gamma[2 * i + idx_1];
					float v2 = init_beta[dev_next_state[i][1] + idx_] + gamma[2 * i + 1 + idx_1];
					sh_beta[0][i] = v1 > v2 ? v1 : v2;
				}
			}
			else if (k > len_this_block - 1 - guard_len && i < n_state)
			{
				int idx = (k + j*sub_block_len) * n_state;
				int idx_ = (k + 1 + j*sub_block_len) * n_state;
				int idx_1 = 2 * n_state * (k + 1 + j*sub_block_len);

				float v1 = sh_beta[len_this_block - 1 - k - 1][dev_next_state[i][0]] + gamma[2 * i + idx_1];
				float v2 = sh_beta[len_this_block - 1 - k - 1][dev_next_state[i][1]] + gamma[2 * i + 1 + idx_1];
				sh_beta[len_this_block - 1 - k][i] = v1 > v2 ? v1 : v2;
				if (j + 1 == n_block)
					beta_out[idx + i] = sh_beta[len_this_block - 1 - k][i];
			}
			else if (k == len_this_block - 1 - guard_len && i < n_state)
			{
				int idx_1 = 2 * n_state * (k + 1 + j*sub_block_len);

				float v1 = sh_beta[guard_len - 1][dev_next_state[i][0]] + gamma[2 * i + idx_1];
				float v2 = sh_beta[guard_len - 1][dev_next_state[i][1]] + gamma[2 * i + 1 + idx_1];
				beta_out[(k + j*sub_block_len) * n_state + i] = v1 > v2 ? v1 : v2;
			}
			else if (i < n_state && k >= 0)
			{
				int idx = (k + j*sub_block_len) * n_state;
				int idx_ = (k + j*sub_block_len + 1) * n_state;
				int idx_1 = 2 * (k + j*sub_block_len + 1)* n_state;
				float v1 = beta_out[dev_next_state[i][0] + idx_] + gamma[2 * i + idx_1];
				float v2 = beta_out[dev_next_state[i][1] + idx_] + gamma[2 * i + idx_1 + 1];
				beta_out[idx + i] = v1 > v2 ? v1 : v2;
			}
			__syncthreads();
		}
	}

}

__global__
void calc_Inp_Out(float* output, float *input, float *sys, float *gamma, float *alpha, float *beta, const int length, const int n_state)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < length)
	{
		int  idx_, idx_1, idx_2, idx_3, idx_4;

		float temp1 = 0;
		float temp2 = 0;

		idx_ = i*n_state;
		idx_1 = 2 * n_state*i;
		idx_2 = idx_ + 1;

		temp1 = alpha[idx_] + gamma[idx_1 + 1] + beta[dev_next_state[0][1] + idx_];
		temp2 = alpha[idx_2] + gamma[idx_1 + 3] + beta[dev_next_state[1][1] + idx_];
		float num = temp1 > temp2 ? temp1 : temp2;

		temp1 = alpha[idx_] + gamma[idx_1] + beta[dev_next_state[0][0] + idx_];
		temp2 = alpha[idx_2] + gamma[idx_1 + 2] + beta[dev_next_state[1][0] + idx_];
		float denum = temp1 > temp2 ? temp1 : temp2;

		for (int jj = 2; jj < n_state; jj++)
		{
			idx_3 = idx_ + jj;
			idx_4 = idx_1 + 2 * jj;

			temp1 = alpha[idx_3] + gamma[idx_4 + 1] + beta[dev_next_state[jj][1] + idx_];
			if (num < temp1)
				num = temp1;

			temp1 = alpha[idx_3] + gamma[idx_4] + beta[dev_next_state[jj][0] + idx_];
			if (denum < temp1)
				denum = temp1;
		}
		output[i] = num - denum;
		input[i] = output[i] - input[i] - sys[i];


	}
}

__global__
void limiter(float* output, float *input, const int len)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < len)
	{
		if (output[i] > 35)
			output[i] = 35;
		else if (output[i] < -35)
			output[i] = -35;
		if (input[i] > 35)
			input[i] = 35;
		else if (input[i] < -35)
			input[i] = -35;
	}

}

__global__
void reorder(float* output, float *input, const int len)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < len)
	{
		output[i] = input[dev_interleaver[i]];
	}

}

__global__
void rereorder(float* output, float *input, const int len)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < len)
	{
		output[dev_interleaver[i]] = input[i];
	}

}

__global__
void calc_out(float* output, float *input, const int len)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if (i < len)
	{
		output[dev_interleaver[i]] = input[i] > 0;
	}
}

__global__
void printdata(float *input, const int length)
{
	for (int i = 0000; i < 10; i++)
	{
		printf("%i %f\n", i, input[i]);
		//printf("%5.6f\n",input[i]);
	}
}


void turbo_dec(float *Data_Deinter, const int fecLen, int* interleaves, float *outData);

int main()
{

	std::string signalLine;
	//------------------------------------------------------//
	std::ifstream signalFile("g:\\Record_data.txt");
	const int nData = 15372;
	float input_data[nData];
	int i = 0;
	if (signalFile.is_open())
	{
		while ((!signalFile.eof()) && (i < nData))
		{
			getline(signalFile, signalLine);
			input_data[i] = atof(signalLine.c_str());
			i++;
		}
		signalFile.close();
	}
	else
		std::cout << "Unable to open signalfile" << std::endl;
	//------------------------------------------------------//
	const int nInterIndex = 5124;
	int *Interleaver_data = new int[nInterIndex];
	std::ifstream interleaverFile("g:\\Interleaver_Index.txt");
	i = 0;
	if (interleaverFile.is_open())
	{
		while ((!interleaverFile.eof()) && (i < nInterIndex))
		{
			getline(interleaverFile, signalLine);
			Interleaver_data[i] = int(atof(signalLine.c_str())) - 1;
			//std::cout << Interleaver_data[i] << " " << i << std::endl;
			i++;
		}
		interleaverFile.close();
	}
	else
		std::cout << "Unable to open interleaverFile" << std::endl;
	//------------------------------------------------------//
	//float Sys1[nData/3],Sys2[nData/3],Parity1[nData/3],Parity2[nData/3];
	float *Data_Deinter = new float[nData];
	int fecLen{ nData / 3 };
	for (int i = 0; i < nData / 3; i++)
	{
		Data_Deinter[i] = input_data[3 * i];
		Data_Deinter[i + fecLen] = input_data[3 * i + 1];
		Data_Deinter[i + 2 * fecLen] = input_data[3 * i + 2];
	}

	std::cout << "Start Processing:" << std::endl;

	float *outData = new float[nData];

	turbo_dec(Data_Deinter, fecLen, Interleaver_data, outData);


	// Add vectors in parallel.

}

void dec_turboWithCuda(int *interleaver, float * Sys1, float * Sys2, float * Parity1, float * Parity2, const int decnum, const int length)
{
	
	for (int chank_now = 0; chank_now < chankk; chank_now++)
	{
		if (chank_now > 0)
		{
			cudaStreamWaitEvent(stream[2], event[3][chank_now - 1], 0);
			cudaStreamWaitEvent(stream[2], event[4][chank_now - 1], 0);
		}
		myfunc << <12, 427, 0, stream[2] >> > (dev_tmp, dev_input_llr, dev_sys1, dev_parity1, length);
		set_Gamma << < data_size, n_state, 0, stream[2] >> > (dev_gamma, dev_input_llr, dev_sys1, dev_parity1, dev_tmp, length, n_state);
		set_AlphaBeta << < blockDimAlphaBeta, n_state, 0, stream[2] >> > (dev_alpha, dev_beta, dev_gamma, dev_init_alpha, dev_init_beta, length, n_subBlock, n_state, guard_len);
		calc_Inp_Out << < ceil(length / 1024.0), 1024, 0, stream[2] >> > (dev_output_llr, dev_input_llr, dev_sys1, dev_gamma, dev_alpha, dev_beta, length, n_state);
		limiter << < ceil(length / 1024.0), 1024, 0, stream[2] >> > (dev_output_llr, dev_input_llr, length);
		reorder << <ceil(length / 1024.0), 1024, 0, stream[2] >> > (dev_input_llr_2, dev_input_llr, length);
		
		cudaEventRecord(event[2][chank_now], stream[2]);

		cudaStreamWaitEvent(stream[3], event[2][chank_now], 0);

		myfunc << <ceil(length / 1024), 1024, 0, stream[3] >> > (dev_tmp, dev_input_llr_2, dev_sys2, dev_parity2, length);
		set_Gamma << < data_size, n_state, 0, stream[3] >> > (dev_gamma, dev_input_llr_2, dev_sys2, dev_parity2, dev_tmp, length, n_state);
		set_AlphaBeta << < blockDimAlphaBeta, n_state, 0, stream[3] >> > (dev_alpha_2, dev_beta_2, dev_gamma, dev_init_alpha_intrlv, dev_init_beta_intrlv, length, n_subBlock, n_state, guard_len);
		calc_Inp_Out << < ceil(length / 1024.0), 1024, 0, stream[3] >> > (dev_output_llr, dev_input_llr_2, dev_sys2, dev_gamma, dev_alpha_2, dev_beta_2, length, n_state);
		limiter << < ceil(length / 1024.0), 1024, 0, stream[3] >> > (dev_output_llr, dev_input_llr_2, length);
		rereorder << < ceil(length / 1024.0), 1024, 0, stream[3] >> > (dev_input_llr, dev_input_llr_2, length);
		
		cudaEventRecord(event[3][chank_now], stream[3]);

		cudaStreamWaitEvent(stream[4], event[2][chank_now], 0);
		cudaStreamWaitEvent(stream[5], event[3][chank_now], 0);

		cudaStatus = cudaMemcpyAsync(dev_init_alpha, dev_alpha, param_size * sizeof(float), cudaMemcpyDeviceToDevice, stream[4]);
		cudaStatus = cudaMemcpyAsync(dev_init_beta, dev_beta, param_size * sizeof(float), cudaMemcpyDeviceToDevice, stream[4]);
		cudaEventRecord(event[4][chank_now], stream[4]);

		cudaStatus = cudaMemcpyAsync(dev_init_alpha_intrlv, dev_alpha_2, param_size * sizeof(float), cudaMemcpyDeviceToDevice, stream[5]);
		cudaStatus = cudaMemcpyAsync(dev_init_beta_intrlv, dev_beta_2, param_size * sizeof(float), cudaMemcpyDeviceToDevice, stream[5]);
		cudaEventRecord(event[5][chank_now], stream[5]);
	}

	calc_out << < ceil(length / 1024.0), 1024, 0, stream[6] >> > (bits_out, dev_output_llr, length);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Elapsed: %f miliseconds\n", elapsedTime);

}

void turbo_dec(float * Data_Deinter, const int fecLen, int * interleaves, float * outData)
{
	float *Sys1 = Data_Deinter;
	float *Parity1 = Data_Deinter + fecLen;
	float *Parity2 = Data_Deinter + 2 * fecLen;
	float *Sys2 = new float[fecLen];
	for (int i = 0; i < fecLen; i++)
	{
		Sys2[i] = Sys1[interleaves[i]];
	}


	float *outtxt;
	dec_turboWithCuda(interleaves, Sys1, Sys2, Parity1, Parity2, 2, fecLen);
	
	return;

}

// Helper function for using CUDA to add vectors in parallel.


