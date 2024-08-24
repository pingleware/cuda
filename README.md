# CUDA Development
To help in understanding CUDA development for NVIDIA GT 730 (https://www.amazon.com/dp/B09KTFWNPK), and is not support under OLLAMA, though this card is supported. I choose this card for it's low profile configuration as the card will be installed in a 2U mini-itx rack case. Since OLLAMA does not support this GPU, a private build of OLLAMA must be performed to include support for this GPU card.

![alt text](images/gpu-windows11.png "GPU Monitorin on Windows 11")


## Developer Environment Configuration

    1. Install GT 730 driver v391.35* from https://www.nvidia.com/Download/driverResults.aspx/132845/en-us/
    2. Install Visual Studio 2022 or latest from https://visualstudio.microsoft.com/downloads/
    3. Install latest CUDA toolkit from https://developer.nvidia.com/cuda-toolkit
    4. Install w64devkit from https://github.com/skeeto/w64devkit/releases 

if installing CUDA Toolkit 10.2.889 from https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal, then

    1. Install Visual Studio 2019
    2. Install CUDA 10.2 Toolkit (link above)
    3. Run setup_cuda_vs2019.bat
    4. Edit C:\Program FIles\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include\crt\host_config.h on line 156, change _MSC_VER >= 1930 to _MSC_VER >= 2000. Administrator access required to edit this file.

*Windows 11 can install the Windows 10 NVIDIA drivers.

## Compile the GPU INFO
Open the Developer POwerdhell for Visual Studio 2022 terminal, then change to the directory and run make to build the gpu_info.exe

```
CUDA-capable device found!
Device Name: Ç
Compute Capability: 865495392.312
Total Global Memory: 134184863 MB
Max Threads per block: 866781026
Clock Rate: 0
PCI Bus ID: 0
PCI Device ID: 0
PCI Domain ID: 2
```

Now that I have a test CUDA application working (detecting the GPU) with this card, progress is being made on the OLLAMA private build.

### Using VS2019 with CUDA Toolkit 10.2

```
CUDA-capable device found!
Device Name: X≈╧Z#
Compute Capability: 0.0
Total Global Memory: 1511403 MB
Max Threads per block: 0
Clock Rate: 35
PCI Bus ID: -760661836
PCI Device ID: 32758
PCI Domain ID: -760345576
```

# Learning Resources

C++ CUDA Tutorial: Theory & Setup: Part 1 - https://www.youtube.com/watch?v=IuxJO0HOcH0

# OpenCL as an alternative
Using the OpenCL-Wrapper, and upgrade the SLN to VS2022, compiled and executed successfully,

```
|----------------.------------------------------------------------------------|
| Device ID    0 | GeForce GT 730                                             |
|----------------'------------------------------------------------------------|
|----------------.------------------------------------------------------------|
| Device ID      | 0                                                          |
| Device Name    | GeForce GT 730                                             |
| Device Vendor  | NVIDIA Corporation                                         |
| Device Driver  | 391.35 (Windows)                                           |
| OpenCL Version | OpenCL C 1.1                                               |
| Compute Units  | 2 at 1400 MHz (384 cores, 1.075 TFLOPs/s)                  |
| Memory, Cache  | 4096 MB, 32 KB global / 48 KB local                        |
| Buffer Limits  | 1024 MB global, 64 KB constant                             |
|----------------'------------------------------------------------------------|
| Info: OpenCL C code successfully compiled.                                  |
| Info: Value before kernel execution: C[0] = 1.00000000                      |
| Info: Value after kernel execution: C[0] = 5.00000000                       |
```

## Building identify_gpu

```
gcc -o identify_gpu.exe identify_gpu.c -I"..\OpenCL-Wrapper\src\OpenCL\include" -L"..\OpenCL-Wrapper\src\OpenCL\lib" -lOpenCL

cl /I"..\OpenCL-Wrapper\src\OpenCL\include" identify_gpu.c /link /LIBPATH:"..\OpenCL-Wrapper\src\OpenCL\lib" OpenCL.lib

```
and the output is

```
Platform 0: NVIDIA CUDA
Device 0: GeForce GT 730
```

## Building matrix_addition

```
cl /I"..\OpenCL-Wrapper\src\OpenCL\include" matrix_addition.cpp -o matrix_add.exe /link /LIBPATH:"..\OpenCL-Wrapper\src\OpenCL\Lib" OpenCL.lib
```

results,
```
Result: 11 22 33 44 55
```
