#include <stdio.h>

const int DSIZE = 8192;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

__global__ void SharedMatrixMul(int* A, int* B, int *C)
{
	__shared__ int As[block_size][block_size]; //	You can't do blockDim.x here since that is not constant and we need to use shared memory dimensions with constants for static initialization.
	__shared__ int Bs[block_size][block_size]; //	For using this, do dynamic memory for shared.

	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idy = blockDim.y*blockIdx.y + threadIdx.y;

	float CValue = 0.0;
	for(int i = 0; i < DSIZE/blockDim.x; ++i)
	{
		// Each thread brings one element of A and B into shared memory. The tiles of A are horizontal and of B are vertical.
		// for (threadIdx.y, threadIdx.x) element of shared memory block, for a particular block, we fill respective elements from A and B.
		// For As, we load one block or tile in horizontal direction first. This is done when i=0. Each thread of the block loads one element of A into As and we get first block in shared memory.
		// Now, i->1 and we load second block in horizontal direction. We need to do this repeatedly because each thread is computing one element of product matrix.
		// Don't think that each block will compute one block of results. For each thread of global index idx and idy, we need the dot product of idx row of A and idy column of B.
		// So, we first load one tile of A (horizontally) into shared memory and one tile of B (vertically) into shared memory and then, next one and so on.
		// With one tile in shared memory, we compute the dot product for threadIdx.x and threadIdx.y respectively for entire tile.
		// Then, we load another tile and compute the dot product of threadIdx.x and threadIdx.y with this new tile and so forth.
		// Where is the benefit, then? Well, when computing matrix multiplication without this, we actually access global memory multiple times for each index. 
		// If we have N*N blocks, we reduce global memory references by a factor of N. Read chapter 5 of Programming Massively Parallel Processors for clarity.

		As[threadIdx.y][threadIdx.x] = A[idy*DSIZE + i*blockDim.x + threadIdx.x];
		Bs[threadIdx.y][threadIdx.x] = B[(i*blockDim.y + threadIdx.y)*DSIZE + idx];

		__syncthreads(); // sync block threads and hence, let all threads write their part of shared memory

		for(int j = 0; j < blockDim.x; ++j)
		{
			CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
			__syncthreads();
		}
	}
	C[idy*DSIZE+idx] = CValue;
}

int main()
{
	int *h_A, *h_B, *h_C;
	int nrOfBytes = DSIZE * DSIZE * sizeof(float);
	h_A = (int*)malloc(nrOfBytes);
	h_B = (int*)malloc(nrOfBytes);
	h_C = (int*)malloc(nrOfBytes);

	for (int i = 0; i < DSIZE*DSIZE; i++)
	{
	    h_A[i] = A_val;
	    h_B[i] = B_val;
	    h_C[i] = 0;
    }

	int *d_A, *d_B, *d_C;
	if(cudaMalloc((void**)&d_A, nrOfBytes) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_B, nrOfBytes) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_C, nrOfBytes) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}


	if(cudaMemcpy(d_A, h_A, nrOfBytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}

	if(cudaMemcpy(d_B, h_B, nrOfBytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}

	dim3 block(block_size, block_size);
	dim3 grid((DSIZE + block_size-1)/(block_size), (DSIZE + block_size-1)/(block_size));
	SharedMatrixMul<<<grid, block>>>(d_A, d_B, d_C);

	if(cudaMemcpy(h_C, d_C, nrOfBytes, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("CUDA Malloc failed\n");
		return 0;
	}	

	for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  	printf("Success!\n"); 
  	return 0;

	return 0;
}