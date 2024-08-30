#include <iostream>
#include <CL/cl.h>

const int arraySize = 5;
const int a[arraySize] = { 1, 2, 3, 4, 5 };
const int b[arraySize] = { 10, 20, 30, 40, 50 };
int c[arraySize] = { 0 };

// OpenCL kernel to add two arrays
const char* kernelSource = R"(
__kernel void addKernel(__global const int* d_a, __global const int* d_b, __global int* d_c, const int size) {
    int i = get_global_id(0);
    if (i < size) {
        d_c[i] = d_a[i] + d_b[i];
    }
}
)";

int main() {
    cl_int err;

    // Step 1: Get platforms and devices
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    err = clGetPlatformIDs(1, &platform_id, nullptr);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);

    // Step 2: Create an OpenCL context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Step 3: Create memory buffers on the device
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, arraySize * sizeof(int), (void*)a, &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, arraySize * sizeof(int), (void*)b, &err);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, arraySize * sizeof(int), nullptr, &err);

    // Step 4: Create and build the OpenCL program and kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "addKernel", &err);

    // Step 5: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err = clSetKernelArg(kernel, 3, sizeof(int), &arraySize);

    // Step 6: Execute the kernel
    size_t globalWorkSize = arraySize;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);

    // Step 7: Read the result back to the host
    err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, arraySize * sizeof(int), c, 0, nullptr, nullptr);

    // Step 8: Print the result
    std::cout << "Result: ";
    for (int i = 0; i < arraySize; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Step 9: Clean up
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
