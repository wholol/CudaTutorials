#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>     
#include <iostream>
#include <algorithm>

__global__ 
void matmul(int* a, int* b, int* c, int N)
{
	int temp = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < N && col < N)
	{
		for (int k = 0; k < N; ++k)
		{
			temp += a[row * N + k] + b[k * N + col];
		}
		c[row * N + col] = temp;
	}
}

void verify_mat_mul(int* a, int* b, int* c, int N)
{
	int temp = 0;
	int max_err = 0;
	for (int row = 0; row < N; ++row)
	{
		for (int col = 0; col < N; ++col)
		{
			for (int k = 0; k < N; ++k)
			{
				temp += a[row * N + k] + b[k * N + col];
			}
			max_err = std::max(0, temp);
		}
	}
	std::cout << "MAX ERROR: " << max_err << std::endl;
}




int main()
{
	int n = 1 << 10;		//2^16

	int bytes = n * n * sizeof(int);
	int* a_host = new int[n * n];
	int* b_host = new int[n * n];
	int* c_host = new int[n * n];

	int* a_device;
	int* b_device;
	int* c_device;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{	
			a_host[i * n + j] = rand() % 100;
			b_host[i * n + j] = rand() % 100;
		}
	}

	cudaMalloc(&a_device, bytes);
	cudaMalloc(&b_device, bytes);
	cudaMalloc(&c_device, bytes);
	cudaMemcpy(a_device, a_host, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b_host, bytes, cudaMemcpyHostToDevice);

	const int numthreads = 32;
	int numblocks = (int)ceil(n / numthreads);

	dim3 blocks(numblocks, numblocks);		//grid size
	dim3 threads(numthreads, numthreads);	//thread per block

	matmul<<<blocks, threads>>> (a_device, b_device, c_device, n);

	cudaDeviceSynchronize();

	cudaMemcpy(c_host, c_device, bytes, cudaMemcpyDeviceToHost);
	
	verify_mat_mul(a_host, c_host, c_host, n);

	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);
}
