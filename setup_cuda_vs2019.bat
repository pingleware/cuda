@echo off

REM Set up Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Set CUDA environment variables
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
set LIB=%CUDA_PATH%\lib\x64;%LIB%
set INCLUDE=%CUDA_PATH%\include;%INCLUDE%

@echo CUDA environment set up for Visual Studio 2019
