
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<iostream>
#include<fstream>
#include<string>

__constant__ int alpha_power[64]{  };

__constant__ int index_gf_element[64]{  };

__constant__ int alpha_synd[2]{ 2, 4 };

typedef int bchInType;

struct BCH_Params {
	int m, n, k, t;
	int n_max, k_max, prim_poly;
	int *gp, *alpha_power;
	int *index_gf_element; 
	int *alpha_synd;
};


__device__ void gf_mul(int *out, int *x1, int *x2, int *alpha_power, int *index, const int n_max, const int len) {
	int alpha_index[4]; //len <= 4
	for (int i{}; i < len; i++) {
		alpha_index[i] = (index[x1[i]] + index[x2[i]] - 2) % n_max;
		out[i] = static_cast<bool>(x1[i]) * static_cast<bool>(x2[i]) * (alpha_power[alpha_index[i]]);
	}
}

__device__ void gf_mul(int *out, int *x1, const int x2, int *alpha_power, int *index, const int n_max, const int len) {
	int alpha_index[4]; //len <= 4
	for (int i{}; i < len; i++) {
		alpha_index[i] = (index[x1[i]] + index[x2] - 2) % n_max;
		out[i] = static_cast<bool>(x1[i]) * static_cast<bool>(x2) * (alpha_power[alpha_index[i]]);
	}
}

__device__ void bitxor(int *out, int *x1, int *x2, const int len) {
	for (int i{}; i < len; i++)
		out[i] = x1[i] ^ x2[i];
}

//dangerous
__device__ void bitxor(int* out, int *x1, const bchInType x2) {
	for (int i{}; i < 2; i++)
		out[i] = x1[i] ^ x2;
}

__device__ void RiBM_BCH_algorithm(int *out, int *syndrome, int *alpha_power, int *index_gf_element, const int n_max) {
	int delta[6]{ syndrome[0], syndrome[1], 0, 1, 0, 0 };
	int theta[4]{ syndrome[1], 0, 1, 0 };
	int gamma{ 1 };
	int gf_mul_out1[4]{}, gf_mul_out2[4]{};
	gf_mul(gf_mul_out1, delta + 2, gamma, alpha_power, index_gf_element, n_max, 4);
	gf_mul(gf_mul_out2, theta, delta[0], alpha_power, index_gf_element, n_max, 4);
	bitxor(delta, gf_mul_out1, gf_mul_out2, 4);
	out[0] = delta[1];
	out[1] = delta[2];
}

//max error_location len? 1
__device__ void chien_search(int *error_location, int *n_error_location, int *lambda, int *alpha_power, int *index_gf_element, const int n_max, const int thd) {
	int alpha_cs[2]{ alpha_power[0], alpha_power[1] };
	int cs_cells[2]{};
	gf_mul(cs_cells, lambda, alpha_power[0], alpha_power, index_gf_element, n_max, 2);
	int index{}, lambda_value[1]{}, bitxor_out[1]{};
	for (int i{}; i < 63; i++) {
		//in-place
		gf_mul(cs_cells, cs_cells, alpha_cs, alpha_power, index_gf_element, n_max, 2);
		lambda_value[0] = 0;
		bitxor(bitxor_out, lambda_value, cs_cells, 1);
		
		bitxor(lambda_value, bitxor_out, cs_cells + 1, 1);
		if (lambda_value[0] == 0) {
			error_location[index] = i;
			index++;
		}
	}
	n_error_location[0] = index;
}

// <<<1, k>>> for the second call
// n_cols = 64 or 63
__global__ void bch_decoder(bchInType *out, bchInType *input, const int n_columns) {
	int i{ threadIdx.x };
	int n_max{ 63 }, n{ 63 }, k{ 57 }, n_cols{ n_columns };
	if ((n_cols == 64 && i >= n) || (n_cols == 63 & i >= k))
		return;

	int syndrome[2]{}, gf_mul_out[2]{};
	for (int cnt{}; cnt < n; cnt++) {
		gf_mul(gf_mul_out, syndrome, alpha_synd, alpha_power, index_gf_element, n_max, 2);
		bitxor(syndrome, gf_mul_out, input[i*n_cols + cnt]);
	}

	int lambda[2]{};
	RiBM_BCH_algorithm(lambda, syndrome, alpha_power, index_gf_element, n_max);
	int lambda_degree{lambda[1] ? 1 : 0};
	
	int error_location[10]{};
	int number_roots[1]{};
	chien_search(error_location, number_roots, lambda, alpha_power, index_gf_element, n_max, i);
	
	int error_location_cnt{};
	if (lambda_degree == number_roots[0]) {
		for (int cnt{}; cnt < k; cnt++) {
			if (error_location[0] == cnt && cnt > 0) { //works because error_location array is sorted (but then i realized its len is 1 wtf)
				out[i*k + cnt] = 1 - input[i*n_cols + cnt]; 
				error_location_cnt++;
			}
			else
				out[i*k + cnt] = input[i*n_cols + cnt];
		}
	}
	else {
		for (int cnt{}; cnt < k; cnt++) 
			out[i*k + cnt] = input[i*n_cols + cnt];
	}
}

//transpose m*n to n*m
__global__ void transpose(bchInType *out, bchInType *in, const int rows, const int cols) {
	int i{ threadIdx.x }, j{ blockIdx.x };
	out[j*cols + i] = in[i*rows + j];
}

//----helpers----//

__global__ void printGPUArr(int* arr, const int len) {
	for (int i{}; i < len; i++)
		printf("%i, ", arr[i]);
	printf("\n");
}

__global__ void printMat(bchInType *arr, const int rows, const int cols, const int rowsToShow, const int colsToShow) {
	for (int i{}; i < rowsToShow; i++) {
		for (int j{}; j < colsToShow; j++)
			printf("%i, ", arr[j*rows + i]);
		printf("\n");
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<typename T>
int readBinFile(T* input, const std::string fileAddress, const int dataLen) {
	std::ifstream inputFile;
	inputFile.open(fileAddress, std::ios::binary);
	if (!inputFile)
	{
		std::cout << " Error, Couldn't find the input file" << "\n";
		return 0;
	}

	int num_elements{};
	inputFile.seekg(0, std::ios::end);
	num_elements = inputFile.tellg() / sizeof(T);
	inputFile.seekg(0, std::ios::beg);
	std::cout << "num of symbols in the file: " << num_elements << ", num of symbols I specified: " << dataLen << std::endl;
	if (dataLen != 0 && dataLen <= num_elements)
		num_elements = dataLen;

	if (inputFile.is_open())
	{
		inputFile.read(reinterpret_cast<char*>(input), sizeof(T) * num_elements);
		inputFile.close();
	}
	else
		std::cout << "Unable to open the input file" << std::endl;

	return num_elements;
}

void recordData(bchInType *input, int sizeOfWrite, std::string fileName)
{
	std::ofstream outFile;
	outFile.open(fileName, std::ios::binary);
	if (outFile.is_open()) {
		std::cout << "isOpen, writing " << sizeOfWrite << " bits to .bin file" << std::endl;
		outFile.write(reinterpret_cast<const char*>(input), sizeof(bchInType)*sizeOfWrite);
		outFile.close();
		std::cout << "ok" << std::endl;
	}
	else
		std::cout << "isNotOpen" << std::endl;
}

int main() {
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return 0;
	}

	int inputLen{ 204801 };
	bchInType *Ch_out_H1 = new bchInType[inputLen];
	int num_elements = readBinFile(Ch_out_H1, "input.bin", inputLen);
	if (inputLen != num_elements)
		inputLen = num_elements;

	const int FrameLengthBit = 4096;
	const int k{ 57 }, n{ 63 };

	bchInType *dev_Frame1, *dev_Frame1_tr;
	cudaMalloc(&dev_Frame1, FrameLengthBit*sizeof(bchInType));
	cudaMalloc(&dev_Frame1_tr, FrameLengthBit * sizeof(bchInType));
	bchInType *dev_chOutDec_col, *dev_chOutDec_col_tr, *dev_chOutDec_row, *dev_chOutDec_row_tr;
	cudaMalloc(&dev_chOutDec_col, sizeof(bchInType) * 57 * 63);
	cudaMalloc(&dev_chOutDec_col_tr, sizeof(bchInType) * 57 * 63);
	cudaMalloc(&dev_chOutDec_row, sizeof(bchInType) * 57 * 57);
	cudaMalloc(&dev_chOutDec_row_tr, sizeof(bchInType) * 57 * 57);

	bchInType *output = new bchInType[57 * 57 * inputLen / FrameLengthBit];
	gpuErrchk(cudaGetLastError());

	for (int i{}; i < inputLen / FrameLengthBit; i++) {
		cudaMemcpy(dev_Frame1, Ch_out_H1 + i*FrameLengthBit, FrameLengthBit*sizeof(bchInType), cudaMemcpyHostToDevice);
		gpuErrchk(cudaGetLastError());

		transpose << <64, 64 >> >(dev_Frame1_tr, dev_Frame1, 64, 64);
		gpuErrchk(cudaGetLastError());

		bch_decoder << <1, 63 >> >(dev_chOutDec_col, dev_Frame1_tr, 64);
		gpuErrchk(cudaGetLastError());

		transpose << <57, 63 >> >(dev_chOutDec_col_tr, dev_chOutDec_col, 57, 63);
		gpuErrchk(cudaGetLastError());

		bch_decoder << <1, 57 >> >(dev_chOutDec_row, dev_chOutDec_col_tr, 63);
		gpuErrchk(cudaGetLastError());

		cudaMemcpy(output + (57 * 57 - 1) * i , dev_chOutDec_row + 1, (57 * 57 - 1) * sizeof(bchInType), cudaMemcpyDeviceToHost);
		gpuErrchk(cudaGetLastError());
	}

	recordData(output, (57 * 57 - 1)*inputLen / FrameLengthBit, "output.bin");

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;
}
