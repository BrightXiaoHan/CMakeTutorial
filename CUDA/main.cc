#include <iostream>
#include <math.h>
#include <chrono>

#include "cudaMatMul.h" 

using namespace std;
using namespace std::chrono;

#define M 10000
#define K 1000
#define N 10000

int main()
{
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

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

    Matrix h_a, h_b, h_c;
    h_a.width = h_a.stride = M; h_a.height = K; h_a.elements = A;
    h_b.width = h_b.stride = K; h_b.height = N; h_b.elements = B;
    h_c.width = h_c.stride = M; h_c.height = N; h_c.elements = C;

    auto startTime = high_resolution_clock::now();
    MatMul(h_a, h_b, h_c);
    auto endTime = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(endTime - startTime); 
    cout << "cuda计算用时" <<double(duration.count()) / 1000000 << "s" << endl;

    free(A);
    free(B);
    free(C);
}