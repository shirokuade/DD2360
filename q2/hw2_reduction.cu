#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <string.h>     
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <cmath> 

#define BLOCK_SIZE 256

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at file %s in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void reduction_kernel(float *input, float *output,  int num_elements) {
 
  //@@ Insert code below to compute reduction using shared memory and atomics

  __shared__ float s_data[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  // 1) Stride sum on each thread
  float my_sum = 0.0f;
  for (; i < num_elements; i += stride) {
    my_sum += input[i];
  }

  s_data[tid] = my_sum;

  __syncthreads();

  // 2) Sum at the block level
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }

  // 3) Sum all the block's sum's and write back to global memory
  if (tid == 0) {
    atomicAdd(output, s_data[0]);
  }
}



int main(int argc, char **argv) {
  
    int inputLength;


  //@@ Insert code below to read in inputLength from args
    if (argc > 1) {
        inputLength = atoi(argv[1]);
    } else {
        inputLength = 1024 * 1024; // Use default
    }
    if (inputLength <= 0) {
      fprintf(stderr, "Error: Invalid input length. Value must be > 0");
      return -1;
    }
    printf("The input length is %d\n", inputLength);
  
    /*@ add other needed data allocation on CPU and GPU here */
    float *hostInput;
    float *deviceInput;
    float *hostOutput;
    float *deviceOutput;
    float *hostRef; 

    size_t inputSize = inputLength * sizeof(float);
    size_t float_size = sizeof(float);

    hostInput = (float *)malloc(inputSize);

    hostOutput = (float *)malloc(float_size);
    hostRef = (float *)malloc(float_size);
    if (!hostInput || !hostOutput || !hostRef) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    CHECK_CUDA_ERROR(cudaMalloc((void**) &deviceInput, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &deviceOutput, float_size));

  
  //@@ Insert code below to initialize the input array with random values on CPU
    const unsigned int seed = 42; 
    std::mt19937 gen(seed); 

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = distribution(gen);
    }

  //@@ Insert code below to create reference result in CPU and add a timer
    auto start_cpu = std::chrono::high_resolution_clock::now();

    *hostRef = 0.0f;
    for (int i = 0; i < inputLength; i++) {
        *hostRef += hostInput[i];
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    printf("CPU Reference Time: %f ms\n", cpu_time.count());

  // GPU Timing events
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

  //@@ Insert code to copy data from CPU to the GPU
    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(deviceOutput, 0, float_size));

  //@@ Initialize the grid and block dimensions here
    int blockSize = 256;
    dim3 dimGrid(ceil((double)inputLength / blockSize), 1, 1); 
    dim3 dimBlock(blockSize, 1, 1);


  //@@ Launch the GPU Kernel here and add a timer
    reduction_kernel<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, inputLength);


  //@@ Copy the GPU memory back to the CPU here
    CHECK_CUDA_ERROR(cudaMemcpy(hostOutput, deviceOutput, float_size, cudaMemcpyDeviceToHost));

    // @@ Stop GPU Timer and get time
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    float gpu_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));
    printf("GPU End-to-End Time: %f ms\n", gpu_time_ms);


  //@@ Insert code below to compare the output with the reference
    float diff = fabsf(*hostRef - *hostOutput);
    float relative_diff = diff / fabsf(*hostRef); // Or use std::max(fabsf(*hostRef), fabsf(*hostOutput))

    // A relative tolerance of 1e-5 (0.001%) is reasonable for this
    float relative_tolerance = 1e-5f; 

    printf("hostRef = %f, hostOutput = %f\n", *hostRef, *hostOutput);

    if (relative_diff < relative_tolerance) {        
        printf("SUCCESS: Same result in CPU and GPU. \n");
    } else {
        printf("FAILURE: DIFFERENT result in CPU and GPU. Difference is %f \n", diff);
    }

  //@@ Free memory here
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    CHECK_CUDA_ERROR(cudaFree(deviceInput));
    CHECK_CUDA_ERROR(cudaFree(deviceOutput));

    free(hostInput);
    free(hostOutput);
    free(hostRef);

    return 0;
}