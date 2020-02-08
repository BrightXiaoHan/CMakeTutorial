#include <cblas.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using namespace std::chrono;

#define M 10000
#define K 1000
#define N 10000

int main()
{
    double *A = new double[M * K];
    double *B = new double[K * N];
    double *C = new double[M * N];

    for (int i = 0; i < M * K; i++)
    {
        A[i] = sin(i);
    }

    for (int i = 0; i < K*N; i++){
        B[i] = cos(i);
    }

    for (int i = 0; i < M * N; i++)
    {
        C[i] = 0.5;
    }

    cublasStatus_t stat; 
    cudaError_t cudaStat; 
    cublasHandle_t handle;               // CUBLAS context

    double *d_A;
    double *d_B;
    double *d_C;
    cudaStat = cudaMalloc((void **)&d_A, M * K * sizeof(*A));
    cudaStat = cudaMalloc((void **)&d_B, K * N * sizeof(*B));
    cudaStat = cudaMalloc((void **)&d_C, M * N * sizeof(*C));

    stat = cublasCreate(&handle); // initialize CUBLAS context

    // copy matrices from the host to the device
    stat = cublasSetMatrix(M, K, sizeof(*A), A, M, d_A, M); //a -> d_a
    stat = cublasSetMatrix(K, N, sizeof(*B), B, K, d_B, K); //b -> d_b
    stat = cublasSetMatrix(M, N, sizeof(*C), C, M, d_C, M); //c -> d_c

    auto startTime = high_resolution_clock::now();
    double alpha = 1.0;
    double beta = 1.0;
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A,
        M, d_B, K, &beta, d_C, M);
    cudaDeviceSynchronize();
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(endTime - startTime); 
    cout << "cblas计算用时" <<double(duration.count()) / 1000000 << "s" << endl;

    stat = cublasGetMatrix(M, N, sizeof(*C), d_C, M, C, M); // cp d_c - >c
    cudaFree(d_A);         // free device memory
    cudaFree(d_B);         // free device memory
    cudaFree(d_C);         // free device memory
    cublasDestroy(handle); // destroy CUBLAS context
    free(A);
    free(B);
    free(C);
}