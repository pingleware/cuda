#include <iostream>
#include <CL/cl.h>

// Function to identify the GPU
void identifyGPU() {
    cl_int err;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;

    // Get the number of platforms
    err = clGetPlatformIDs(0, nullptr, &platformCount);
    platforms = new cl_platform_id[platformCount];
    err = clGetPlatformIDs(platformCount, platforms, nullptr);

    if (platformCount == 0) {
        std::cout << "No OpenCL platforms found." << std::endl;
        return;
    }

    for (cl_uint i = 0; i < platformCount; ++i) {
        // Get the number of devices on the platform
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (deviceCount == 0) {
            std::cout << "No OpenCL-capable devices found on platform " << i << "." << std::endl;
            continue;
        }

        devices = new cl_device_id[deviceCount];
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, nullptr);

        for (cl_uint j = 0; j < deviceCount; ++j) {
            char deviceName[128];
            cl_uint computeUnits;
            cl_ulong globalMem;
            size_t maxWorkGroupSize;
            cl_uint clockFrequency;
            cl_uint pciBusID = 0, pciDeviceID = 0;

            // Get device name
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
            // Get number of compute units
            err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            // Get global memory size
            err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr);
            // Get max work group size
            err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
            // Get clock frequency
            err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFrequency), &clockFrequency, nullptr);

            // For OpenCL, PCI bus ID and device ID might be available as extensions
            // They are often vendor-specific, and may not be available universally

            std::cout << "OpenCL-capable device found!" << std::endl;
            std::cout << "Device Name: " << deviceName << std::endl;
            std::cout << "Compute Units: " << computeUnits << std::endl;
            std::cout << "Total Global Memory: " << globalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Max Work Group Size: " << maxWorkGroupSize << std::endl;
            std::cout << "Clock Frequency: " << clockFrequency << " MHz" << std::endl;
            std::cout << "PCI Bus ID: " << pciBusID << std::endl;
            std::cout << "PCI Device ID: " << pciDeviceID << std::endl;
            std::cout << std::endl;
        }
        delete[] devices;
    }
    delete[] platforms;
}

int main() {
    identifyGPU();
    return 0;
}
