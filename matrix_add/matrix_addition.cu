#include <iostream>
#include <cuda_runtime.h>

const int arraySize = 5;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
int c[arraySize] = { 0 };

// CUDA kernel to add two arrays
__global__ void addKernel(const int* d_a, const int* d_b, int* d_c, int size) {
    int i = threadIdx.x;

    if (i < size) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main() {
    // Define device pointers
    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to add arrays on the GPU
    addKernel<<<1, arraySize>>>(d_a, d_b, d_c, arraySize);

    // Copy result back from device (GPU) to host (CPU)
    cudaMemcpy(c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
