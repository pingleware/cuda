#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main() {
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    char buffer[1024];
    
    // Get the number of platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);

    // Get all platforms
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (cl_uint i = 0; i < platformCount; i++) {
        // Print platform information
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
        printf("Platform %d: %s\n", i, buffer);

        // Get the number of devices available on this platform
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);

        // Get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

        for (cl_uint j = 0; j < deviceCount; j++) {
            // Print device information
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
            printf("  Device %d: %s\n", j, buffer);
        }

        free(devices);
    }

    free(platforms);
    return 0;
}
