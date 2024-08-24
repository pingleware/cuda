// main.cu
#include <iostream>
#include "gpu_info.cuh"

// Function to identify the GPU
void identifyGPU() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        std::cout << "CUDA-capable device found!" << std::endl;
        std::cout << "Device Name: " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    }
}

int main() {
    identifyGPU();
    return 0;
}
