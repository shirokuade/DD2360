#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <string.h>     
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>
#include <cmath> 
#define NUM_BINS 4096

// CUDA error checking macro
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at file %s in line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

    // 1. Create histogram in shared memory
    __shared__ unsigned int s_bins[NUM_BINS];
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        s_bins[i] = 0;

    }
    __syncthreads();

    // 2. Use a grid-stride loop tocompute local block histograms
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = global_idx; i < num_elements; i += total_threads) {
        unsigned int bin_index = input[i];
        
        atomicAdd(&s_bins[bin_index], 1U);
    }

    __syncthreads();

    // 3. Add local block histograms to global histogram 

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], s_bins[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if(i<num_bins){
    if(bins[i] > 127) bins[i] = 127;
  } 

}


int main(int argc, char **argv) {
 
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    int distribution_option;

    // GPU Timing events
    cudaEvent_t start_gpu, stop_gpu;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_gpu));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    // @@ Read in inputLength from args
    if (argc > 1) {
        inputLength = atoi(argv[1]);
    } else {
        inputLength = 1024 * 1024; // Use as default value
    }
    
    distribution_option = 2; // Use a default value
    if (argc > 2) {
        distribution_option = atoi(argv[2]);
    }

    printf("The input length is %d\n", inputLength);
 
    // @@ Allocate Host memory for input and output
    size_t inputSize = inputLength * sizeof(unsigned int);
    size_t binsSize = NUM_BINS * sizeof(unsigned int);

    hostInput = (unsigned int *)malloc(inputSize);
    hostBins = (unsigned int *)malloc(binsSize);
    resultRef = (unsigned int *)malloc(binsSize);

    if (!hostInput || !hostBins || !resultRef) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // @@ Initialize hostInput to random numbers
    const unsigned int seed = 42; 
    std::mt19937 gen(seed); 

    if (distribution_option == 1) {
        std::uniform_int_distribution<> distribution(0, NUM_BINS - 1);
        for (int i = 0; i < inputLength; i++) {
            hostInput[i] = static_cast<unsigned int>(distribution(gen));
        }
    } else if (distribution_option == 2) {
        std::normal_distribution<> distribution((NUM_BINS - 1) / 2.0, (NUM_BINS - 1) / 6.0);
        for (int i = 0; i < inputLength; i++) {
            int val = static_cast<int>(distribution(gen));
            if (val < 0) val = 0;
            if (val >= NUM_BINS) val = NUM_BINS - 1;
            hostInput[i] = static_cast<unsigned int>(val);
        }
    } else {
        fprintf(stderr, "Error: Invalid distribution_option %d\n", distribution_option);
        free(hostInput); free(hostBins); free(resultRef);
        exit(-1);
    }

    // @@ Create reference result in CPU (with <chrono> timing)
    auto start_cpu = std::chrono::high_resolution_clock::now();

    memset(resultRef, 0, binsSize); // Initialize result to 0
    
    for (int i = 0; i < inputLength; i++) {
        resultRef[hostInput[i]]++;
    }

    for (int i = 0; i < NUM_BINS; i++) {
        if (resultRef[i] > 127) {
            resultRef[i] = 127;
        }
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    printf("CPU Reference Time: %f ms\n", cpu_time.count());


    // @@ Allocate GPU memory here
    CHECK_CUDA_ERROR(cudaMalloc((void**) &deviceInput, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**) &deviceBins, binsSize));

    // @@ Copy memory to the GPU here
    CHECK_CUDA_ERROR(cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice));

    
    // @@ Start GPU Timer
    CHECK_CUDA_ERROR(cudaEventRecord(start_gpu));


    // @@ Initialize GPU results
    CHECK_CUDA_ERROR(cudaMemset(deviceBins, 0, binsSize));

    // @@ Initialize the grid and block dimensions here
    int blockSize1 = 1024; 
    
    int numBlocks1 = 128; 
    
    dim3 dimGrid1(numBlocks1, 1, 1); 
    dim3 dimBlock1(blockSize1, 1, 1);


    // @@ Launch the GPU Kernel here
    histogram_kernel<<<dimGrid1, dimBlock1>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
    // @@ Initialize the second grid and block dimensions here
    int blockSize2 = 256;
    dim3 dimGrid_2(ceil((double)NUM_BINS / blockSize2), 1, 1);
    dim3 dimBlock_2(blockSize2, 1, 1);

    // @@ Launch the second GPU Kernel here
    convert_kernel<<<dimGrid_2, dimBlock_2>>>(deviceBins, NUM_BINS);

    // @@ Stop GPU Timer and get time
    CHECK_CUDA_ERROR(cudaEventRecord(stop_gpu));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_gpu));

    // @@ Copy the GPU memory back to the CPU here
    CHECK_CUDA_ERROR(cudaMemcpy(hostBins, deviceBins, binsSize, cudaMemcpyDeviceToHost));


    float gpu_time_ms = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));
    printf("GPU End-to-End Time: %f ms\n", gpu_time_ms);



    // @@ Compare the output with the reference
    bool sameResult = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != resultRef[i]) {
            printf("Mismatch at bin %d: CPU=%u, GPU=%u\n", i, resultRef[i], hostBins[i]);
            sameResult = false;
        }
    }

    if (sameResult) {
        printf("SUCCESS: Same result in CPU and GPU. \n");
    } else {
        printf("FAILURE: DIFFERENT result in CPU and GPU. \n");
    }

    // Write output to csv
    FILE *csv_file = fopen("histogram_output.csv", "w");
    if (csv_file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
    } else {
        fprintf(csv_file, "Bin,Count\n");
        
        for (int i = 0; i < NUM_BINS; i++) {
            fprintf(csv_file, "%d,%u\n", i, hostBins[i]);
        }
        
        fclose(csv_file);
    }

    
    // @@ Free the GPU memory here
    CHECK_CUDA_ERROR(cudaFree(deviceInput));
    CHECK_CUDA_ERROR(cudaFree(deviceBins));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_gpu));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_gpu));

    // @@ Free the CPU memory here
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}