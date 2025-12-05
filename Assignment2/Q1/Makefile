NVCC       := nvcc
TARGET     := hw2_histogram
SRC        := hw2_histogram.cu
NVCC_FLAGS := -O3

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
