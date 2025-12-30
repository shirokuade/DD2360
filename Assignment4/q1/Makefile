NVCC       := nvcc
TARGET     := hw4-convolution
SRC        := hw4-convolution.cu
NVCC_FLAGS := -arch=sm_75

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)