#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

/* Our over-simplified CUDA kernel */
/* Parallel vector add kernel: c[i] = a[i] + b[i] for i < n */
__global__ void add(const int *a, const int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char **argv) {
    // Problem size (can be overridden by command-line)
    int N = 1 << 20; // default 1,048,576 elements
    if (argc > 1) {
        int v = atoi(argv[1]);
        if (v > 0) N = v;
    }

    size_t bytes = (size_t)N * sizeof(int);

    // Allocate in host memory and initialize
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2*i;
        h_c[i] = 0;
    }

    // Allocate in device memory
    int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_c, bytes));

    // Invoke the CUDA kernel with GPU timing using CUDA events
    cudaEvent_t start_evt, stop_evt;
    CHECK(cudaEventCreate(&start_evt));
    CHECK(cudaEventCreate(&stop_evt));
    CHECK(cudaEventRecord(start_evt, 0));


    // Copy from host memory to device memory
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Initialize thread block and thread grid
    int tpb = 128;
    int blocks = (N + tpb - 1) / tpb;

    // Print launch configuration
    long long total_threads = (long long)blocks * (long long)tpb;
    long long extra_threads = total_threads - (long long)N;
    printf("Configuration: N=%zu, blocks=%d, threads_per_block=%d, total_threads=%lld, extra_threads=%lld\n",
        (size_t)N, blocks, tpb, total_threads, extra_threads);

    add<<<blocks, tpb>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop_evt, 0));
    CHECK(cudaEventSynchronize(stop_evt));

    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, start_evt, stop_evt));

    // Copy result from GPU to CPU
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventDestroy(start_evt));
    CHECK(cudaEventDestroy(stop_evt));

    // Compute CPU reference with timing
    int *h_ref = (int*)malloc(bytes);
    if (!h_ref) {
        fprintf(stderr, "Host malloc for reference failed\n");
        return 1;
    }
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        h_ref[i] = h_a[i] + h_b[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_d = cpu_end - cpu_start;
    double cpu_s = cpu_d.count();

    // Compare results
    long long mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_ref[i]) {
            ++mismatches;
            if (mismatches <= 10) {
                printf("mismatch at %d: gpu=%d cpu=%d\n", i, h_c[i], h_ref[i]);
            }
        }
    }
    if (mismatches == 0) {
        printf("All %d results match. OK\n", N);
    } else {
        printf("Found %lld mismatches out of %d\n", mismatches, N);
    }

    // Print timings and throughput
    double gpu_s = (double)gpu_ms / 1000.0;
    double bytes_processed = 3.0 * (double)N * sizeof(int); // read a, read b, write c
    double gpu_gb_s = bytes_processed / (gpu_s * 1e9);
    double cpu_gb_s = bytes_processed / (cpu_s * 1e9);
    double gpu_melems_s = (double)N / (gpu_s * 1e6);
    double cpu_melems_s = (double)N / (cpu_s * 1e6);
    printf("Execution Time: %.3f ms, throughput: %.3f GB/s (%.3f Melem/s)\n", gpu_ms, gpu_gb_s, gpu_melems_s);
    //printf("CPU reference time: %.3f ms, throughput: %.3f GB/s (%.3f Melem/s)\n", cpu_s*1000.0, cpu_gb_s, cpu_melems_s);

    free(h_ref);

    // Cleanup host and device memory
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return (mismatches == 0) ? 0 : 2;
}
