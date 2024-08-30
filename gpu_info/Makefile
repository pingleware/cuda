# Makefile for CUDA project

# Compiler
NVCC = nvcc

# Target executable
TARGET = gpu_info

# Source files
SRC = main.cu

# Compiler flags (optional)
NVCC_FLAGS =

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Phony targets
.PHONY: all
