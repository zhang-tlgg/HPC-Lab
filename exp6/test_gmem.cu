#include <cstdio>
#include <cuda.h>
#include <iostream>

// You should modify this parameter.
// #define STRIDE 2

__global__ void stride_copy(float *dst, float *src, int STRIDE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i * STRIDE] = src[i * STRIDE];
}

int main() {
    for (int STRIDE = 1; STRIDE <= 8; STRIDE *= 2){
        float *dev_a = 0, *dev_b = 0;
        int size = (1 << 24);
        cudaMalloc((void **)&dev_a, size  * 32 * sizeof(float));
        cudaMalloc((void **)&dev_b, size  * 32 * sizeof(float));
        dim3 gridSize(size / 1024, 1);
        dim3 blockSize(1024, 1);

        cudaEvent_t st, ed;
        cudaEventCreate(&st);
        cudaEventCreate(&ed);
        float duration;

        // The parameters that you should change.
        for (int t = 0; t < 1024; t++) {
            stride_copy<<<gridSize, blockSize>>>(dev_b, dev_a, STRIDE);
        }
        cudaEventRecord(st, 0);
        for (int t = 0; t < 1024; t++) {
            stride_copy<<<gridSize, blockSize>>>(dev_b, dev_a, STRIDE);
        }
        cudaEventRecord(ed, 0);
        cudaEventSynchronize(st);
        cudaEventSynchronize(ed);
        cudaEventElapsedTime(&duration, st, ed);
        duration /= 1024;
        std::cout << "stride:    " << STRIDE << std::endl;
        std::cout << "bandwidth: " << 8 * size / duration / 1e6 << std::endl;
        cudaFree(dev_a);
        cudaFree(dev_b);
    }
}

