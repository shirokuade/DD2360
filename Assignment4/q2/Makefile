NVCC       := nvcc
NVCC_FLAGS := -arch=sm_75
CUBLAS_FLAGS := -lcublas -lcusparse

TARGETS := hw4-heat hw4-heat_nopref
SRCS    := hw4-heat.cu hw4-heat_nopref.cu

all: $(TARGETS)

hw4-heat: hw4-heat.cu
	$(NVCC) $(NVCC_FLAGS) -o hw4-heat hw4-heat.cu $(CUBLAS_FLAGS)

hw4-heat_nopref: hw4-heat_nopref.cu
	$(NVCC) $(NVCC_FLAGS) -o hw4-heat_nopref hw4-heat_nopref.cu $(CUBLAS_FLAGS)

clean:
	rm -f $(TARGETS)