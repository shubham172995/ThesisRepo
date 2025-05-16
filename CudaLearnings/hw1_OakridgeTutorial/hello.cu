#include<iostream>

__global__ void helloKernel()
{
	printf("Hello from block: %u, thread: %u\n",blockIdx.x, threadIdx.x);
}

int main()
{
	helloKernel<<<2,2>>>();
	cudaDeviceSynchronize();
	return 0;
}