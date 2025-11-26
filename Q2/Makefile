# Makefile for two independent CUDA programs

NVCC       := nvcc
NVCC_FLAGS := -O3

TARGETS := hw2_reduction hw2_reduction_multiple
SRCS    := hw2_reduction.cu hw2_reduction_multiple.cu

all: $(TARGETS)

hw2_reduction: hw2_reduction.cu
	$(NVCC) $(NVCC_FLAGS) -o hw2_reduction hw2_reduction.cu

hw2_reduction_multiple: hw2_reduction_multiple.cu
	$(NVCC) $(NVCC_FLAGS) -o hw2_reduction_multiple hw2_reduction_multiple.cu

clean:
	rm -f $(TARGETS)
