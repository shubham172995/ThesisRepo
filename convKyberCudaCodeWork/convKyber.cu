#include <stdio.h>
#include <cstdlib>

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


/*
    Use chrono for timing results
*/

 
// BELOW IS A SKELETON CODE ON HOW TO USE ABOVE MACRO. Every CUDA runtime API call returns an error code.
// It's good practice to rigorously check these error codes

const int DSIZE = 32; // Say

/*
    When we implement three phase conv_kyber, we might not use it. Also, for now, lets work with this batch size
    to have good structure as is in the conv_kyber's paper.
*/
const int TWO_PHASE_BATCH_SIZE = 8; 

/*
    Initializing Kyber-Parameters for Kyber-512, Kyber-768, and Kyber-1024 respectively.
*/
const int KYBER_512_n     = 256;
const int KYBER_512_k     = 2;
const int KYBER_512_q     = 3329;
const int KYBER_512_eta1  = 3;
const int KYBER_512_eta2  = 2;

const int KYBER_768_n     = 256;
const int KYBER_768_k     = 3;
const int KYBER_768_q     = 3329;
const int KYBER_768_eta1  = 2;
const int KYBER_768_eta2  = 2;

const int KYBER_1024_n    = 256;
const int KYBER_1024_k    = 4;
const int KYBER_1024_q    = 3329;
const int KYBER_1024_eta1 = 2;
const int KYBER_1024_eta2 = 2;



/*
void ntt(int16_t r[256]) {
  unsigned int len, start, j, k;
  int16_t t, zeta;

  k = 1;
  for(len = 128; len >= 2; len >>= 1) {
    for(start = 0; start < 256; start = j + len) {
      zeta = zetas[k++];
      for(j = start; j < start + len; j++) {
        t = fqmul(zeta, r[j + len]);
        r[j + len] = r[j] - t;
        r[j] = r[j] + t;
      }
    }
  }
}
*/

/*
    These are Montgommery reduced values of powers of zeta. Also, Mont(x) = x.R mod q. This table represents
    for each index i from 0 to 127, Montgommery of zeta^(bitRev7(i));
    Now, to get MontGommery(zeta^{2*bitRev7(i) + 1}), we need Montgommery(zeta^{bitRev(i)}), square it
    and then, we need Montgommery(zeta^{bitRev7(i)}) such that bitRev7(i) gives 1.
    Now, 7 bit reversal of 64 gives 1. So, we need 64th entry of the table for this.
    So, to get MontGommery(zeta^{2*bitRev7(i) + 1}), do squaring i^th entry and multiply with i=64 (i.e., 65th entry) of the table.

    zetas[i] = Mont(zeta^(bitrev7(i))) for i in [0..127]  (Mont(x) = x * R mod q).
    To get Mont(zeta^(2*bitrev7(i) + 1)):
      1) t = mont_mul(zetas[i], zetas[i]);    // Mont(zeta^(2*bitrev7(i)))
      2) result = mont_mul(t, zetas[64]);     // zetas[64] = Mont(zeta^1) because bitrev7(64) == 1
    Thus: result == Mont(zeta^(2*bitrev7(i) + 1))
*/
const int16_t zetasInMontgommeryReducedForm[128] = {
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
};
/*
__device__ void ntt(int16_t* f)
{
    u_int8_t len;
    u_int16_t start, j, k, zeta, t;

    for(len = 128; len >= 2; ++len)
    {
        for(start = 0; start < 256; start = j+len)
        {
            zeta = zetas[k++];
            for(j = start; j < start + len; ++j)
            {
                t = zeta * f[j+len]; // USE MODULAR MULTIPLICATION HERE.
                f[j+len] = f[j] - t;
                f[j] = f[j] + t;
            }
        }
    }
}
*/

/*
    Arguments - 
        f0     - vector of N/2 (128 for Kyber) coefficients of polynomial. 0 stands for even index coefficients.
        f1     - vector of N/2 (128 for Kyber) coefficients of polynomial. 1 stands for odd index coefficients.
        fHat0  - vector of N/2 (128 for Kyber) coefficients of NTT of polynomial. 0 stands for even index coefficients.
        fHat1  - vector of N/2 (128 for Kyber) coefficients of NTT of polynomial. 1 stands for odd index coefficients.
        bitRev - vector of 16 elements which contains bitreversal for all numbers from 1 to 15.
*/


/*
__device__ void ntt(u_int16_t* f0, u_int16_t* f1, u_int16_t* fHat0, u_int16_t* fHat1, u_int8_t* bitRev)
{
    u_int8_t i0, i1, j0, j1;
    for(i0 = 0; i0 < 16; ++i0)
    {
        for(i1 = 0; i1 < 8; ++i1)
        {
            for(j0 = 0; j0 < 16; ++j0)
            {
                for(j1 = 0; j1 < 8; ++j1)
                {
                    fHat0[16*bitRev[i0] + 2*bitRev[i1]]     += f0[16*bitRev[j0] + 2*bitRev[j1]];
                    fHat1[16*bitRev[i0] + 2*bitRev[i1] + 1] += f1[16*bitRev[j0] + 2*bitRev[j1] + 1];
                }
            }
        }
    }
}
*/

//  i = 8i0 + i1

//__global__ void BatchedNTT(u_int8_t* bitRevArray, int16_t* polyCoeffs, int16_t* polyCoeffs_0, int16_t* polyCoeffs_1)
__global__ void BatchedNTT(u_int8_t* bitRevArray, int16_t* polyCoeffs_0, int16_t* polyCoeffs_1)

{
    int idx  = threadIdx.x; //  Just 1 block for now. So, thread Idx is global Idx as of now.

    __shared__ int16_t reconstructionMatrices[8][16][16];

    /*
        Each warp works on one reconstruction matrix. So, divide the index by 32 to get the matrix
        that this warp threads need to work on.
        Each reconstruction matrix is 16*16. Hence, 32 threads will work on 256 elements in parallel.
        So, 2 threads per row. Hence, one thread retrieves 8 elements of shared memory since matrix is 16*16.
        So, even thread index works on 8 elements of polyCoeffs_0 row and odd threads on 8 elements of polyCoeffs_1 row.

        So, first index is for matrix index. Hence, divide by 32 to get warp index that works on that matrix.
        Second index is row index. Two threads per row. So, take last 5 bits which are for row (0-31) and divide by 2 to get 0-15.
        So, idx is 32 bits and each hex represents 4 bits. Now, I want least significant 5 bits which represent
        the thread index in the warp. So, first take them and then, since it is two threads per row, divide by 2.
        Last index is column index. So, loop from 0 to 7 for each thread retrieving the respective row.
        
        Now, when accessing from the array, based on whether the thread index is even or odd, we access the even array or odd array.
        For this, just see whether the index is even or odd. Based on that, access the elements of matrix.
    */

   u_int8_t matrixIndex, rowIndex;
   matrixIndex = idx >> 5;
   rowIndex = ((idx & 0x0000001f) >> 1);
   for(int i = 0; i < 8; ++i)
   {
    u_int16_t indexOfRow = (matrixIndex * (KYBER_512_n >> 1) + rowIndex * 8 + i);
        if(idx & 1)
        {
            
            reconstructionMatrices[matrixIndex][rowIndex][(i << 1)] = polyCoeffs_0[indexOfRow];
        }
        else
        {
            reconstructionMatrices[matrixIndex][rowIndex][((i << 1) | 1)] = polyCoeffs_1[indexOfRow];
        }
   }
    __syncthreads();

    /*
        Now, check once whether these are to be transposed. If yes, make appropriate changes.
    */
}


/*************************************************************************************HOST CODE BEGINS************************************************************************************/


/*
    Function Description -
    This function takes one array, h_bitRevArray of 16 entries. For each value from 0 to 15, h_bitRevArray will hold the bit reversal
    of each of this value. This is done to facilitate the splitting of the index during NTT computation. This will be done on CPU 
    and the resultant arrays will be passed to GPU.

    Arguments - 
        h_bitRevArray - An array of 16 8-bit unsigned integers. Upon returning, it will hold bit reversal of each of the 4-bit unsigned value from 0 to 15.

    The naming is done as per the index naming convention in ConvKyber paper when they split the 128 entry array in 16*8 array and we need these
    reverse indices for powers of zeta factors.
*/

inline void BitRev(u_int8_t* h_bitRevArray)
{
    for(u_int8_t i = 0; i < 16; ++i)
    {
        u_int16_t iRev = 0;
        while(i)
        {
            iRev = (iRev << 1) | (i & 1);
            i >>= 1;
        }
        h_bitRevArray[i] = iRev;
    }
}

/*
    ****************************************************** TO DO ******************************************************

    1. Use better randomness here.
    2. Inlined it for now. Should be more performant than normal call is.
*/

inline void InitializePolynomialCoefficients(int16_t* polyCoeffs)
{
    for(int i = 0; i < KYBER_512_n; ++i)
    {
        polyCoeffs[i] = rand() % 3329;
    }
}


int main(){

    /*
        ****************************************************** TO DO ******************************************************

        Check if you want to initialize the host arrays using heap or stack suffices. Maybe, compare the performances of the two
    */

   /*
        ****************************************************** TO DO ******************************************************

        Incorporate Fusion Kernels as well.
    */

    /*  
        GET THE bit reversal array. For each number in 0-15, h_bitRevArray holds the reverse bits of it.
        The naming is done as per the naming convention used in ConvKyber paper.
    */  

    u_int8_t h_bitRevArray[16], *d_bitRevArray;
    BitRev(h_bitRevArray);

    /*
        The bit reversal for each entry is done. 
    */

    /*
        Initialize the polynomials in the field. Using 2 phase batching for now.
        Also, using parameters of Kyber-512 for now. Made them 16 bit signed ints since sign will be useful at the
        time of Montgommery reduction


        ****************************************************** TO DO ******************************************************

        Make this modular to use correct parameters. Use an argument from Command Line and use
        it to select appropriate parameters.
    */

    int16_t h_polyCoeffs[TWO_PHASE_BATCH_SIZE][KYBER_512_n], h_polyCoeffs_0[TWO_PHASE_BATCH_SIZE][KYBER_512_n >> 1], h_polyCoeffs_1[TWO_PHASE_BATCH_SIZE][KYBER_512_n >> 1];
    int16_t *d_polyCoeffs, *d_polyCoeffs_0, *d_polyCoeffs_1;

    for(u_int8_t i = 0; i < TWO_PHASE_BATCH_SIZE; ++i)
    {
        InitializePolynomialCoefficients(h_polyCoeffs[i]);   
    }

    /*
        Split the 256 entry vector (polynomial coefficients) into two vectors, f0 and f1, which
        contain even and odd coefficients of polynomial respectively
    */

    for(u_int8_t polyIndex = 0; polyIndex < TWO_PHASE_BATCH_SIZE; ++polyIndex)
    {
        for(int i = 0; i < KYBER_512_n; ++i)
        {
            if(i & 1)
            {
                h_polyCoeffs_1[polyIndex][i >> 1] = h_polyCoeffs[polyIndex][i];
            }
            else
            {
                h_polyCoeffs_0[polyIndex][i >> 1] = h_polyCoeffs[polyIndex][i];
            }
        }
    }

    /*
        Splitting done
    */

    /*
        Allocate device memory and copy input data over to GPU
    */

    //cudaError_t errorInCudaCall;

    cudaMalloc((void**)& d_bitRevArray, 16*sizeof(u_int8_t));
    cudaMalloc((void**)& d_polyCoeffs, TWO_PHASE_BATCH_SIZE*KYBER_512_n*sizeof(int16_t));
    cudaMalloc((void**)& d_polyCoeffs_0, TWO_PHASE_BATCH_SIZE*(KYBER_512_n >> 1)*sizeof(int16_t));
    cudaMalloc((void**)& d_polyCoeffs_1, TWO_PHASE_BATCH_SIZE*(KYBER_512_n >> 1)*sizeof(int16_t));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_bitRevArray, h_bitRevArray, 16*sizeof(u_int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polyCoeffs, h_polyCoeffs, TWO_PHASE_BATCH_SIZE*KYBER_512_n*sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polyCoeffs_0, h_polyCoeffs_0, TWO_PHASE_BATCH_SIZE*(KYBER_512_n >> 1)*sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polyCoeffs_1, h_polyCoeffs_1, TWO_PHASE_BATCH_SIZE*(KYBER_512_n >> 1)*sizeof(int16_t), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    /*
        Launch kernel
    */

    dim3 block(TWO_PHASE_BATCH_SIZE * 32); //   Using 1 warp per precomputed matrix multiplication
    //BatchedNTT<<<1, block>>>(d_bitRevArray, d_polyCoeffs, d_polyCoeffs_0, d_polyCoeffs_1);  //  Using just 1 block
    BatchedNTT<<<1, block>>>(d_bitRevArray, d_polyCoeffs_0, d_polyCoeffs_1);  //  Using just 1 block

    cudaCheckErrors("kernel launch failure");

    return 0;
}