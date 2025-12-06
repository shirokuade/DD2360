#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

// Define the number of streams we will use for concurrency
#define NUM_STREAMS 4
// Default segment size: 128MB per segment (32M elements)
#define S_SEG_DEFAULT (32 * 1024 * 1024)

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
    int S_seg = S_SEG_DEFAULT;

    if (argc > 1) {
        int v = atoi(argv[1]);
        if (v > 0) N = v;
    }
    if (argc > 2) {
        int v = atoi(argv[2]);
        if (v > 0) S_seg = v;
    }
    if (S_seg > N) S_seg = N; // Ensure segment size doesn't exceed N

    size_t bytes_total = (size_t)N * sizeof(int);
    size_t bytes_seg = (size_t)S_seg * sizeof(int);

    int num_segments = (N + S_seg - 1) / S_seg;

    // --- 1. Allocate PINNED Host Memory ---
    // Pinned memory is required for cudaMemcpyAsync to be truly asynchronous.
    int *h_a, *h_b, *h_c;
    CHECK(cudaHostAlloc((void**)&h_a, bytes_total, cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void**)&h_b, bytes_total, cudaHostAllocDefault));
    // h_c only needs to store the final result
    CHECK(cudaHostAlloc((void**)&h_c, bytes_total, cudaHostAllocDefault));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
        h_c[i] = 0; // Clear result buffer
    }

    // --- 2. Allocate Device Memory (4 buffers for concurrent processing) ---
    // We allocate 4 smaller device buffers, one for each stream's active segment.
    int *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        // Only allocate segment size (S_seg) in the device buffers
        CHECK(cudaMalloc((void **)&d_a[i], bytes_seg));
        CHECK(cudaMalloc((void **)&d_b[i], bytes_seg));
        CHECK(cudaMalloc((void **)&d_c[i], bytes_seg));
    }

    // --- 3. Create CUDA Streams ---
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK(cudaStreamCreate(&streams[i]));
    }

    // --- Setup Timing Events ---
    cudaEvent_t start_evt, stop_evt;
    CHECK(cudaEventCreate(&start_evt));
    CHECK(cudaEventCreate(&stop_evt));

    printf("\nStream: Configuration: N=%d, Segment Size (S_seg)=%d, Num Segments=%d, Num Streams=%d\n",
           N, S_seg, num_segments, NUM_STREAMS);

    // --- START CONCURRENT EXECUTION ---
    CHECK(cudaEventRecord(start_evt, 0));

    // --- 4. Asynchronous Loop for Segmented Processing ---
    // We cycle through segments, assigning work to streams in a round-robin fashion.
    for (int i = 0; i < num_segments; ++i) {
        int stream_idx = i % NUM_STREAMS; // Cycle stream indices: 0, 1, 2, 3, 0, 1, ...

        // Calculate the starting index for this segment
        size_t offset = (size_t)i * S_seg;

        // Calculate the actual number of elements in this segment (last segment may be smaller)
        int seg_N = (i == num_segments - 1) ? (N - offset) : S_seg;
        size_t seg_bytes = (size_t)seg_N * sizeof(int);

        // --- 4a. Asynchronous Host-to-Device Copy (H2D) ---
        // Copies h_a[offset] into d_a[stream_idx] on the specified stream.
        CHECK(cudaMemcpyAsync(d_a[stream_idx], h_a + offset, seg_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));
        CHECK(cudaMemcpyAsync(d_b[stream_idx], h_b + offset, seg_bytes, cudaMemcpyHostToDevice, streams[stream_idx]));

        // --- 4b. Launch Kernel (Computation) ---
        int tpb = 128;
        int blocks = (seg_N + tpb - 1) / tpb;

        // Kernel launch is inherently asynchronous. It uses the specified stream.
        add<<<blocks, tpb, 0, streams[stream_idx]>>>(d_a[stream_idx], d_b[stream_idx], d_c[stream_idx], seg_N);
        CHECK(cudaGetLastError()); // Check for kernel launch errors immediately

        // --- 4c. Asynchronous Device-to-Host Copy (D2H) ---
        // Copies d_c[stream_idx] back into h_c[offset] on the specified stream.
        CHECK(cudaMemcpyAsync(h_c + offset, d_c[stream_idx], seg_bytes, cudaMemcpyDeviceToHost, streams[stream_idx]));
    }
    //

    // --- 5. Synchronize All Streams ---
    // The host must wait for ALL operations across ALL streams to finish.
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK(cudaStreamSynchronize(streams[i]));
    }

    CHECK(cudaEventRecord(stop_evt, 0));
    CHECK(cudaEventSynchronize(stop_evt));

    // --- 6. Print Timings and Throughput ---
    float gpu_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_ms, start_evt, stop_evt));

    // Check a few spots to ensure the result is non-zero (i.e., the kernel ran)
    int validation_error = 0;
    if (h_c[N / 2] != (N / 2) + 2 * (N / 2)) {
        printf("Validation error: h_c[%d] = %d, expected %d\n", N / 2, h_c[N / 2], 3 * (N / 2));
        validation_error = 1;
    } else {
        printf("Quick check at index %d: h_c[%d] = %d. Seems OK.\n", N / 2, N / 2, h_c[N / 2]);
    }

    double gpu_s = (double)gpu_ms / 1000.0;
    // Data processed: 2 inputs (h_a, h_b) + 1 output (h_c). All transfers are overlapping.
    double bytes_transferred_total = 3.0 * bytes_total;
    double gpu_gb_s = bytes_transferred_total / (gpu_s * 1e9);
    double gpu_melems_s = (double)N / (gpu_s * 1e6);

    printf("Execution Time: %.3f ms, Throughput: %.3f GB/s (%.3f Melem/s)\n", gpu_ms, gpu_gb_s, gpu_melems_s);

    // --- 7. Cleanup ---
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK(cudaFree(d_a[i]));
        CHECK(cudaFree(d_b[i]));
        CHECK(cudaFree(d_c[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));

    CHECK(cudaEventDestroy(start_evt));
    CHECK(cudaEventDestroy(stop_evt));

    return (validation_error == 0) ? 0 : 2;
}
