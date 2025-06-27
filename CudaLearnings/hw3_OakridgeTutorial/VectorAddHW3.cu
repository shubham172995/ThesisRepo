#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 32*1048576;
// vector add kernel: C = A + B
__global__ void vadd(const float *A, const float *B, float *C, int ds){

  for (int idx = threadIdx.x+blockDim.x*blockIdx.x; idx < ds; idx+=gridDim.x*blockDim.x)         // a grid-stride loop
    C[idx] = A[idx]+B[idx];         // do the vector (element) add here
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;}
  cudaMalloc(&d_A, DSIZE*sizeof(float));  // allocate device space for vector A
  cudaMalloc(&d_B, sizeof(float)*DSIZE); // allocate device space for vector B
  cudaMalloc(&d_C, sizeof(float)*DSIZE); // allocate device space for vector C
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy vector A to device:
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  // copy vector B to device:
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //cuda processing sequence step 1 is complete

  int blocks = 160;  // modify this line for experimentation
  int threads = 1024; // modify this line for experimentation
  // This second variable must be constrained to choices between 1 and 1024, inclusive. These are limits imposed by the GPU hardware. i.e., threads variable for number of threads in a block

  vadd<<<blocks, threads>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector C from device to host:
  cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  printf("A[0] = %f\n", h_A[33]);
  printf("B[0] = %f\n", h_B[33]);
  printf("C[0] = %f\n", h_C[33]);
  return 0;
}

/*
    RESULTS WITH 1 BLOCK AND 1 THREAD PER BLOCK
    *******************************************

    vadd(const float *, const float *, float *, int) (1, 1, 1)x(1, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- --------------
    Metric Name             Metric Unit   Metric Value
    ----------------------- ----------- --------------
    DRAM Frequency                  Ghz           5.99
    SM Frequency                    Ghz           1.46
    Elapsed Cycles                cycle  4,346,985,180
    Memory Throughput                 %           0.34
    DRAM Throughput                   %           0.34
    Duration                          s           2.97
    L1/TEX Cache Throughput           %           4.63
    L2 Cache Throughput               %           0.14
    SM Active Cycles              cycle 217,347,451.45
    Compute (SM) Throughput           %           0.23
    ----------------------- ----------- --------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Mbyte/s       648.83
    Mem Busy                               %         0.14
    Max Bandwidth                          %         0.34
    L1/TEX Hit Rate                        %        90.62
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        38.22
    Mem Pipes Busy                         %         0.23
    ---------------------------- ----------- ------------
*/

/*
    RESULTS WITH 1 BLOCK AND 1024 THREADS PER BLOCK
    ***********************************************

    vadd(const float *, const float *, float *, int) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.46
    Elapsed Cycles                cycle   19,592,507
    Memory Throughput                 %        16.35
    DRAM Throughput                   %        16.35
    Duration                         ms        13.40
    L1/TEX Cache Throughput           %        42.86
    L2 Cache Throughput               %         4.41
    SM Active Cycles              cycle   978,537.50
    Compute (SM) Throughput           %         1.61
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.1 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s        31.34
    Mem Busy                               %         4.41
    Max Bandwidth                          %        16.35
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        34.02
    Mem Pipes Busy                         %         1.61
    ---------------------------- ----------- ------------
*/

/*
    RESULTS WITH 160 BLOCKS AND 1024 THREADS PER BLOCK
    **************************************************

    vadd(const float *, const float *, float *, int) (160, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.46
    Elapsed Cycles                cycle    3,280,775
    Memory Throughput                 %        93.70
    DRAM Throughput                   %        93.70
    Duration                         ms         2.24
    L1/TEX Cache Throughput           %        18.64
    L2 Cache Throughput               %        25.28
    SM Active Cycles              cycle 3,245,948.55
    Compute (SM) Throughput           %         9.58
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s       179.57
    Mem Busy                               %        25.28
    Max Bandwidth                          %        93.70
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        33.33
    Mem Pipes Busy                         %         9.58
    ---------------------------- ----------- ------------
*/