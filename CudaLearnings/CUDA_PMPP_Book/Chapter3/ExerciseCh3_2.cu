#include<iostream>

#define N 1024
#define BLOCK_DIM 1024

bool CheckCorrectness(int *Parallel, int *Serial)
{
	for(int i = 0; i < N; ++i)
	{
		if(Parallel[i] != Serial[i])
		{
			return false;
		}
	}
	return true;
}

void __global__ MatrixMul(int *B, int *C, int *A)
{
	int idx = blockDim.x*blockIdx.x + threadIdx.x;

	for(int k = 0; k < N; ++k)
	{
		A[idx] += B[idx*N + k] * C[k];
	}
}

void InitializeMatrixVector (int *B, int *C, int *A, int *tempRes)
{
	for(int i = 0; i < (N*N); ++i)
	{
		B[i] = i%2 ? i : i*i;
	}

	for(int i = 0; i < N; ++i)
	{
		C[i] = i%3 ? i : i*(i+1);
		A[i] = 0;
		tempRes[i] = 0;
	}
}

int main(){
	int *h_A, *h_B, *h_C, *tempRes, *d_A, *d_B, *d_C;
	int nrOfBytesMatrix = sizeof(int)*N*N;
	int nrOfBytesVector = sizeof(int)*N;


	h_B = (int *) malloc(nrOfBytesMatrix);
	h_C = (int *) malloc(nrOfBytesVector);
	h_A = (int *) malloc(nrOfBytesVector);
	tempRes = (int *) malloc(nrOfBytesVector);

	InitializeMatrixVector (h_B, h_C, h_A, tempRes);

	if(cudaMalloc((void**)&d_B, nrOfBytesMatrix) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_C, nrOfBytesVector) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_A, nrOfBytesVector) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMemcpy(d_B, h_B, nrOfBytesMatrix, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Host to Device memory copy failed\n");
		return 0;
	}

	if(cudaMemcpy(d_C, h_C, nrOfBytesVector, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Host to Device memory copy failed\n");
		return 0;
	}

	dim3 blockSize(BLOCK_DIM, 1, 1);
	dim3 gridSize(((N-1)/BLOCK_DIM +1), 1, 1);

	MatrixMul<<<gridSize, blockSize>>>(d_B, d_C, d_A);

	if(cudaMemcpy(h_A, d_A, nrOfBytesVector, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Device to Host memory copy failed\n");
		return 0;
	}

	for(int i = 0; i < N; ++i)
	{
		for(int j = 0; j < N; ++j)
		{
			tempRes[i] += h_B[i*N + j] * h_C[j];
		}
	}

	bool res = CheckCorrectness(h_A, tempRes);

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
