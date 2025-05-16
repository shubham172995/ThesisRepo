#include<iostream>
#define N (2048*2048)
#define NUM_THREADS_PER_BLOCK 512

__global__ void add(int* a, int* b, int* c)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < N)
		c[index] = a[index] + b[index];
}

void initialize(int*a)
{
	for(int i = 0; i < N; ++i)
	{
		a[i] = i%2 ? i:i/2;
	}
}

int main()
{
	int *a, *b, *c, *d_a, *d_b, *d_c;

	a = (int*)malloc(N*sizeof(int));
	b = (int*)malloc(N*sizeof(int));
	c = (int*)malloc(N*sizeof(int));

	int nrOfBytes = N*sizeof(int);

	initialize(a);
	initialize(b);

	if(cudaMalloc((void**)&d_a, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_b, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	if(cudaMalloc((void**)&d_c, nrOfBytes) != cudaSuccess)
	{
		printf("GPU memory init failed\n");
		return 0;
	}

	cudaMemcpy(d_a, a, nrOfBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, nrOfBytes, cudaMemcpyHostToDevice);


	add<<<(N+NUM_THREADS_PER_BLOCK-1)/(NUM_THREADS_PER_BLOCK),NUM_THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

	cudaDeviceSynchronize();

	cudaMemcpy(c, d_c, nrOfBytes, cudaMemcpyDeviceToHost);

	bool flag = true;
	for(int i = 0; i < N; i++)
	{
		if(c[i] != a[i] + b[i])
		{
			flag = false;
		}
	}

	if(!flag)
	{
		printf("failed\n");
	}
	else
	{
		printf("passed\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(a);
	free(b);
	free(c);

	return 0;
}