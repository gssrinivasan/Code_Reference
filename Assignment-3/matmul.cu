/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <iostream>
#include <exception>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "cublas_v2.h"
using namespace std;

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C, REAL *C_base);
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C, REAL *C_base);
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C, REAL *C_base);

//kernel code for kernel version 1-global memory
__global__ void cuda_vanilla(int N, REAL *A, REAL *B, REAL *C )
{
int row= blockIdx.y*blockDim.y+threadIdx.y;
int col= blockIdx.x*blockDim.x+threadIdx.x;
if((row<N)&&(col<N))
   {
    float sum = 0.0;
    #pragma unroll
    for(int i=0;i<N;++i)
       {
       sum += A[row*N+i]*B[col+i*N];
       }
    __syncthreads();
    C[row*N+col] = sum;   
   }
}

//kernel code for kernel version 2-shared memory
template <int BLOCK_SIZE> __global__ void cuda_shmem (float *C, float *A, float *B, int wA, int wB)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;

 
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int aBegin = wA * BLOCK_SIZE * by;


    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;

  
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/* main function*/
int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_openmp; /* for timing */

    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*4);

    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];
    REAL *C_v1 = (REAL *) malloc(N*N*4);
    REAL *C_v2 = (REAL *) malloc(N*N*4);
    REAL *C_v3 = (REAL *) malloc(N*N*4);
    
    srand48((1 << 12));
    init(N, N, A);
    init(N, N, B);
    
    cudaSetDevice(0);
        
    elapsed_base = read_timer();
    matmul_base(N, A, B, C_base);
    elapsed_base = (read_timer() - elapsed_base);
       
    elapsed_openmp = read_timer();
    matmul_openmp(N, A, B, C_openmp, num_tasks);
    elapsed_openmp = (read_timer() - elapsed_openmp);

    printf("\n\n\n======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    printf("matmul_GPU:-------------------------------------------------------------------------------------------------\n");      
    matmul_cuda_v1_vanilla(N, A, B, C_v1, C_base);//call to version 1 function    
    matmul_cuda_v2_shmem(N,A,B,C_v2,C_base);//call to version 2 function   
    matmul_cuda_v3_cublas(N,A,B,C_v3,C_base);//call to version 3 function   
    printf("------------------------------------------------------------------------------------------------------------\n\n\n");
    free(heap_buffer);
    return 0;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

/*
 * call to version kernel function that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C_v1, REAL *C_base)
{

   int mem_size_A = sizeof(float)*N*N;
   int mem_size_B = sizeof(float)*N*N;
   int mem_size_C = sizeof(float)*N*N;
   float *d_A, *d_B, *d_C;
   cudaMalloc((void **) &d_A, mem_size_A);
   cudaMalloc((void **) &d_B, mem_size_B);
   cudaMalloc((void **) &d_C, mem_size_C);
   cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
   dim3 dimgrid ((N-1)/16+1,(N-1)/16+1,1);
   dim3 dimblock (16,16,1);
  
   // Performs warmup operation using kernel
   cuda_vanilla <<<dimgrid,dimblock>>> (N, d_A, d_B, d_C);
   cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, NULL);
    
    // Execute the kernel - acessing global memory
    cuda_vanilla <<<dimgrid,dimblock>>> (N, d_A, d_B, d_C);

    // Record the stop event
    cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    // Computing the performance
    double elapsed_cuda_v1 = msecTotal;//time
    double flopsPerMatrixMul_v1 = 2.0 *N*N*N;
    double gigaFlops_v1 = (flopsPerMatrixMul_v1 * 1.0e-9f) / (elapsed_cuda_v1 / 1000.0f);//performance
   
    // Copy result from device to host
   cudaMemcpy(C_v1, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    
    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    
    printf("GlobalMemory_V1:\t%4f\t%4fx10^3 \t\t\t%g\n", elapsed_cuda_v1, gigaFlops_v1, maxerror(N, N, C_base, C_v1));
}

/*
 * call to kernel version 2 that use GPU shared memory
 */
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C_v2, REAL *C_base)
{
    
    int block_size = 16;
    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);
    dimsA.x = N;
    dimsA.y = N;
    dimsB.x = N;
    dimsB.y = N;
        
    int mem_size_A = sizeof(float)*N*N;
    int mem_size_B = sizeof(float)*N*N;
    int mem_size_C = sizeof(float)*N*N;
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, mem_size_A);
    cudaMalloc((void **) &d_B, mem_size_B);
    cudaMalloc((void **) &d_C, mem_size_C);
    cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);
    
    // Performs warmup operation using matrixMul CUDA kernel
    cuda_shmem<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    cudaDeviceSynchronize(); 
    
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, NULL); 
    
    // Executing the kernel
    cuda_shmem<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
   
    // Record the stop event
    cudaEventRecord(stop, NULL);
    
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    
    // Computing the performance
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double elapsed_cuda_v2 = msecTotal;//time
    double flopsPerMatrixMul_v2 = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops_v2 = (flopsPerMatrixMul_v2 * 1.0e-9f) / (elapsed_cuda_v2 / 1000.0f);//performance
    
    // Copy result from device to host
    cudaMemcpy(C_v2, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    
    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();
    
    printf("SharedMemory_V2:\t%4f\t%4fx10^3\t\t\t%g\n", elapsed_cuda_v2, gigaFlops_v2, maxerror(N, N, C_base, C_v2));
}
 
/*
 * call to sgemm of cublas library 
 */

void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C_v3, REAL *C_base) 
{
   int mem_size_A = sizeof(float)*N*N;
   int mem_size_B = sizeof(float)*N*N;
   int mem_size_C = sizeof(float)*N*N;
   float *d_A, *d_B, *d_C;
   cudaMalloc((void **) &d_A, mem_size_A);
   cudaMalloc((void **) &d_B, mem_size_B);
   cudaMalloc((void **) &d_C, mem_size_C);
   cudaMemcpy(d_A, A, mem_size_A, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, B, mem_size_B, cudaMemcpyHostToDevice);
   
   //initialize the cublas parameters
    const float alpha = 1.0f;
    const float beta  = 0.0f;
  
    // Create the cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    //Perform warmup operation with cublas
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, NULL);
    
    // Execute the matrix-vector multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    
    // Record the stop event
    cudaEventRecord(stop, NULL);
    
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    
    // Computing the performance
    double elapsed_cuda_v3 = msecTotal;//time
    double flopsPerMatrixMul_v3 = 2.0 *N*N*N;
    double gigaFlops_v3 = (flopsPerMatrixMul_v3 * 1.0e-9f) / (elapsed_cuda_v3 / 1000.0f);//performance
   
    // Copy result from device to host
    cudaMemcpy(C_v3, d_C, mem_size_C, cudaMemcpyDeviceToHost);
    
    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaDeviceReset();
    printf("cuBLAS_V3:\t\t%4f\t%4fx10^3 \t\t\t%g\n", elapsed_cuda_v3, gigaFlops_v3, maxerror(N, N, C_base, C_v3));
    
    
}