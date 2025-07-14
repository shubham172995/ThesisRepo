#include<iostream>

#define N 1024
#define BLOCK_DIM 1024

bool CheckCorrectness(int *Parallel, int *Serial)
{
	for(int i = 0; i < N; ++i)
	{
		for(int j = 0; j < N; ++j)
		{
			if(Parallel[i*N+j] != Serial[i*N+j])
			{
				return false;
			}
		}
	}
	return true;
}

void __global__ MatrixMul(int *A, int *B, int *C)
{
	int col = blockDim.x*blockIdx.x + threadIdx.x;

	for(int row = 0; row < N; ++row)
	{
		int sum = 0;
		for(int k = 0; k < N; ++k)
		{
			sum += A[row*N + k] * B[k*N + col];
		}
		C[row*N + col] = sum;
	}
}

void InitializeMatrices (int *A, int *B, int *h_tempProduct)
{
	for(int i = 0; i < (N*N); ++i)
	{
		A[i] = i%2 ? i : i*i;
		B[i] = i%3 ? i : i*(i+1);
		h_tempProduct[i] = 0;
	}	
}

int main(){
	int *h_A, *h_B, *h_C, *h_tempProduct, *d_A, *d_B, *d_C;
	int nrOfBytes = sizeof(int)*N*N;

	h_A = (int *) malloc(nrOfBytes);
	h_B = (int *) malloc(nrOfBytes);
	h_C = (int *) malloc(nrOfBytes);
	h_tempProduct = (int *) malloc(nrOfBytes);

	InitializeMatrices (h_A, h_B, h_tempProduct);

	if(cudaMalloc((void**)&d_A, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_B, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_C, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMemcpy(d_A, h_A, nrOfBytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Host to Device memory copy failed\n");
		return 0;
	}

	if(cudaMemcpy(d_B, h_B, nrOfBytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Host to Device memory copy failed\n");
		return 0;
	}

	dim3 blockSize(BLOCK_DIM, 1, 1);
	dim3 gridSize(((N-1)/BLOCK_DIM +1), 1, 1);

	MatrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C);

	if(cudaMemcpy(h_C, d_C, nrOfBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Device to Host memory copy failed\n");
		return 0;
	}

	for(int i = 0; i < N; ++i)
	{
		for(int j = 0; j < N; ++j)
		{
			for(int k = 0; k < N; ++k)
			{
				h_tempProduct[i*N + j] += h_A[i*N + k] * h_B[k*N + j];
			}
		}
	}	

	bool res = CheckCorrectness(h_C, h_tempProduct);

	if(res)
	{
		printf("Correct\n");
	}

	else
	{
		printf("Wrong\n");
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
