#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>

#define gpuCheck(stmt)                                                       \
  do {                                                                       \
      cudaError_t err = stmt;                                                \
      if (err != cudaSuccess) {                                              \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                             \
      }                                                                      \
  } while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define TILE_WIDTH 256

__global__ void convolution_1D_basic(float *N, float *M, float *P, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Width) {
        float Pvalue = 0;
        int N_start_point = i - (MASK_WIDTH / 2);
        for (int j = 0; j < MASK_WIDTH; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
                Pvalue += N[N_start_point + j] * M[j];
            }
        }
        P[i] = Pvalue;
    }
}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int Width)
{
  int tx = threadIdx.x;
  int output_width = TILE_WIDTH - MASK_WIDTH + 1;
  int i_out = blockIdx.x * output_width + tx;
  int i_in = i_out - (MASK_WIDTH / 2);

  __shared__ float input_tile[TILE_WIDTH];
  
  if (i_in >= 0 && i_in < Width) {
      input_tile[tx] = N[i_in];
  } else {
      input_tile[tx] = 0.0f;
  }
  
  __syncthreads();

  if (tx < output_width && i_out < Width) {
      float Pvalue = 0.0f;
      for (int j = 0; j < MASK_WIDTH; j++) {
          Pvalue += input_tile[tx + j] * M[j];
      }
      P[i_out] = Pvalue;
  }
}

//@ Function to verify that convolution is correct
void verify_convolution(float *N, float *M, float *P, int Width) {
    for (int i = 0; i < Width; i++) {
        float Pvalue = 0;
        int N_start_point = i - (MASK_WIDTH / 2);
        for (int j = 0; j < MASK_WIDTH; j++) {
            if (N_start_point + j >= 0 && N_start_point + j < Width) {
                Pvalue += N[N_start_point + j] * M[j];
            }
        }
        if (abs(P[i] - Pvalue) > 1e-3) {
            printf("Verification has failed at index %d! GPU: %f, CPU: %f\n", i, P[i], Pvalue);
            return;
        }
    }
    printf("Verification Paased!\n");
}
int main(int argc, char *argv[]) {
 
  // Read the arguments from the command line
  int N = atoi(argv[1]);


  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N

  cputimer_start();
  //@@ Allocate the host memory
  hostN = (float*)malloc(N * sizeof(float));
  hostM = (float*)malloc(MASK_WIDTH * sizeof(float));
  hostP = (float*)malloc(N * sizeof(float));
  cputimer_stop("Allocated host memory");


  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();

  //@@ Allocate the device memory
  gpuCheck(cudaMalloc((void**)&deviceN, N * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceM, MASK_WIDTH * sizeof(float)));
  gpuCheck(cudaMalloc((void**)&deviceP, N * sizeof(float)));
  cputimer_stop("Allocated device memory");

  
  cputimer_start();
  //@@ Initialize N with random values
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  //@@ Initialize P with 0.0
  for (int i = 0; i < N; i++) hostN[i] = (float)rand() / (float)(RAND_MAX);
  hostM[0] = -0.25; hostM[1] = 0.5; hostM[2] = 1.0; hostM[3] = 0.5; hostM[4] = 0.25;
  for (int i = 0; i < N; i++) hostP[i] = 0.0;
  cputimer_stop("Initialized random values");

  
  cputimer_start();
  gpuCheck(cudaMemcpy(deviceN, hostN, N * sizeof(float), cudaMemcpyHostToDevice));
  gpuCheck(cudaMemcpy(deviceM, hostM, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");
  

  /* Call the basic kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (basic)

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  convolution_1D_basic<<<blocksPerGrid, threadsPerBlock>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(basic)");
  
  cputimer_start();
  gpuCheck(cudaMemcpy(hostP, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");
 
  cputimer_start();
  verify_convolution(hostN, hostM, hostP, N);
  cputimer_stop("Verifying basic GPU convolution");

  /* Call the tiled kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)

  int out_elements = TILE_WIDTH - MASK_WIDTH + 1;
  int tiledBlocks = (N + out_elements - 1) / out_elements;
  convolution_1D_tiled<<<tiledBlocks, TILE_WIDTH>>>(deviceN, deviceM, deviceP, N);
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Finished 1D convolution(tiled)");
  
  cputimer_start();
  gpuCheck(cudaMemcpy(hostP, deviceP, N * sizeof(float), cudaMemcpyDeviceToHost));
  cputimer_stop("Copying output P to the CPU and print out the results");


  //@@ Validate the results from the two implementations
  // Validation is done against cpu version after running each function
  
  cputimer_start();
  verify_convolution(hostN, hostM, hostP, N);
  cputimer_stop("Verifying tiled GPU convolution");


  cputimer_start();
  free(hostN); 
  free(hostM); 
  free(hostP);
  cudaFree(deviceN); 
  cudaFree(deviceM); 
  cudaFree(deviceP);
  cputimer_stop("Free memory resources");

  return 0;
}