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

const size_t DSIZE = 16384;      // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024

// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds){

  int idx = blockIdx.x*blockDim.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[idx*ds + i];         // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}

// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds){

  int idx = blockIdx.x*blockDim.x + threadIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds){
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
      sum += A[i*ds + idx];         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
}}

bool validate(float *data, size_t sz){
  for (size_t i = 0; i < sz; i++)
    if (data[i] != (float)sz) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], (float)sz); return false;}
    return true;
}

int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
    
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
    
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float)); // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
    
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
    
  //cuda processing sequence step 1 is complete
  row_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
    
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("row sums correct!\n");
    
  cudaMemset(d_sums, 0, DSIZE*sizeof(float));
    
  column_sums<<<(DSIZE+block_size-1)/block_size, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
    
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
    
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("column sums correct!\n");
  return 0;
}


//      NSIGHT COMPUTE PERFORMANCE ANALYSIS RESULTS

/*
    sudo <Path To ncu> --section SpeedOfLight --section MemoryWorkloadAnalysis ./matrix_sumHW3
    ******************************************************************************************

    row_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- -------------
    Metric Name             Metric Unit  Metric Value
    ----------------------- ----------- -------------
    DRAM Frequency                  Ghz          5.99
    SM Frequency                    Ghz          1.46
    Elapsed Cycles                cycle    33,013,637
    Memory Throughput                 %         40.67
    DRAM Throughput                   %         29.99
    Duration                         ms         22.57
    L1/TEX Cache Throughput           %         48.35
    L2 Cache Throughput               %         23.77
    SM Active Cycles              cycle 27,765,064.05
    Compute (SM) Throughput           %          2.54
    ----------------------- ----------- -------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.5 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s        57.47
    Mem Busy                               %        40.67
    Max Bandwidth                          %        29.99
    L1/TEX Hit Rate                        %        81.24
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %        21.78
    Mem Pipes Busy                         %         2.54
    ---------------------------- ----------- ------------

  column_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz         5.99
    SM Frequency                    Ghz         1.46
    Elapsed Cycles                cycle    8,390,821
    Memory Throughput                 %        97.75
    DRAM Throughput                   %        97.75
    Duration                         ms         5.74
    L1/TEX Cache Throughput           %        20.03
    L2 Cache Throughput               %        26.35
    SM Active Cycles              cycle 8,375,517.05
    Compute (SM) Throughput           %        10.00
    ----------------------- ----------- ------------

    INF   This workload is utilizing greater than 80.0% of the available compute or memory performance of the device.   
          To further improve performance, work will likely need to be shifted from the most utilized to another unit.   
          Start by analyzing DRAM in the Memory Workload Analysis section.                                              

    Section: Memory Workload Analysis
    ---------------------------- ----------- ------------
    Metric Name                  Metric Unit Metric Value
    ---------------------------- ----------- ------------
    Memory Throughput                Gbyte/s       187.33
    Mem Busy                               %        26.35
    Max Bandwidth                          %        97.75
    L1/TEX Hit Rate                        %            0
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Compression Input Sectors      sector            0
    L2 Hit Rate                            %         0.02
    Mem Pipes Busy                         %        10.00
    ---------------------------- ----------- ------------
*/

/*
    sudo <Path To ncu> --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sumHW3
    *******************************************************************************************************************************************

    row_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                8,388,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector  268,162,682
    ----------------------------------------------- ----------- ------------

  column_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                8,388,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   33,554,432
    ----------------------------------------------- ----------- ------------
*/

//  WE CAN SEE THAT COLUMN SUM IS FASTER AND HAS BETTER MEMORY THROUGHPUT. IF YOU LOOK AT THE CODE, IN ROW SUM,
//  EACH THREAD WORKS ON CONTIGUOUS MEMORY LOCATIONS. SO, FOR A WARP, YOU ACCESS 32 DIFFERENT ROWS AND HENCE,
//  THERE ARE A LOT OF MEMORY ACCESSES. IN COLUMN, YOU LOAD A LINE IN CACHE AND EACH THREAD OF A WARP BENEFITS FROM ONE LOAD.

//  EVEN WHEN CACHING SHOULD HELP IN ROW SUM SINCE EACH ROW THREAD SHOULD BRING A DIFFERENT ROW INTO CACHE, SUBSEQUENT
//  ACCESSES SHOULD BE CACHED FOR EACH ROW BUT CACHE SIZE WON'T BE ABLE TO ACCOMODATE ALL ROWS FOR THE BLOCKS SCHEDULED
//  ON A SINGLE SM. SO, EVEN WHEN CACHING SHOULD HELP, IT WON'T HELP THAT MUCH SINCE THERE WILL BE OVERWRITES.