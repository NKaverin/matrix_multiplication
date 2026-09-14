#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s in %s at line %d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__
void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    printf("%d\n", gridDim.x);
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    int size = n * sizeof(float);
    float* A_d, * B_d, * C_d;

    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);


    // operation
    vecAddKernel << <ceil(n / 256.0), 256 >> > (A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int n = 1024;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> distrib(1, 100);
    float* A_h = (float*)malloc(sizeof(float) * n);
    float* B_h = (float*)malloc(sizeof(float) * n);
    float* C_h = (float*)malloc(sizeof(float) * n);
    for (size_t i = 0; i < n; i++) {
        A_h[i] = distrib(gen);
        B_h[i] = distrib(gen);
    }
    helloFromGPU << <1, 5 >> > ();
    vecAdd(A_h, B_h, C_h, n);
    cudaDeviceSynchronize();
    free(A_h);
    free(B_h);
    free(C_h);

}
