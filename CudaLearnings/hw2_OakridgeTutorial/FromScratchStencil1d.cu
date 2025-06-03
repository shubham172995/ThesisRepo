#include <stdio.h>

#define RADIUS 16
#define BLOCK_SIZE 256
#define N (2048*2048)

void initializeStencil(int *input)
{
	for(int i=0; i<N; ++i)
	{
		input[i] = i%2?i:i+2;
	}
}

__global__ void Stencil1D(int* input, int* output)
{
  __shared__ int temp[BLOCK_SIZE+2*RADIUS];

  temp[threadIdx.x + RADIUS] = input[blockDim.x*blockIdx.x + threadIdx.x];

  if(threadIdx.x < RADIUS)
  {
    temp[threadIdx.x] = input[blockDim.x*blockIdx.x + threadIdx.x - RADIUS];
    temp[blockDim.x + threadIdx.x + RADIUS] = input[blockDim.x*(blockIdx.x + 1) + threadIdx.x];
  }

  __syncthreads();

  int result = 0;
  for(int i = threadIdx.x; i <= threadIdx.x+(2*RADIUS); i++)
  {
    result+= input[i];
  }

  output[blockDim.x*blockIdx.x + threadIdx.x] = result;
}
int main()
{
	int *h_input, *h_output, *d_input, *d_output;
	int nrOfBytes = N*sizeof(int);

	h_input = (int*)malloc(nrOfBytes);
	h_output = (int*)malloc(nrOfBytes);

	if(cudaMalloc((void**)&d_input, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_output, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	initializeStencil(h_input);

	cudaMemcpy(d_input, h_input, nrOfBytes, cudaMemcpyHostToDevice);

	Stencil1D<<<(N+BLOCK_SIZE-1)/(BLOCK_SIZE), BLOCK_SIZE>>>(d_input, d_output);

	cudaMemcpy(h_output, d_output, nrOfBytes, cudaMemcpyDeviceToHost);

	return 0;
}