// burn.cu
// Simple GPU load generator: runs a busy compute kernel repeatedly for a
// specified duration, printing timestamps so we can correlate with an
// external power-polling script.
//
// Usage: burn.exe <seconds_to_run> <intensity 1-10>
//
// "Intensity" controls how much work each kernel launch does (via a loop
// count inside the kernel), which lets you generate different sustained
// power-draw levels for telemetry-lag testing.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define N (1 << 22)  // ~4M elements

__global__ void burn_kernel(float* data, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = data[idx];
    // Busy-work: repeated fused multiply-adds to keep ALUs occupied.
    for (int i = 0; i < iters; ++i) {
        x = x * 1.0000001f + 0.0000001f;
        x = x - 0.0000001f * x;
    }
    data[idx] = x;
}

static long long now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <seconds_to_run> <intensity 1-10>\n", argv[0]);
        return 1;
    }
    double seconds = atof(argv[1]);
    int intensity = atoi(argv[2]);
    if (intensity < 1) intensity = 1;
    if (intensity > 10) intensity = 10;
    int iters = intensity * 20000; // tune this if your card is too fast/slow to saturate

    float* h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fprintf(stdout, "PHASE0_START %lld\n", now_ms());
    fflush(stdout);

    auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    while (elapsed < seconds) {
        burn_kernel<<<blocks, threads>>>(d_data, iters);
        cudaDeviceSynchronize();
        elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
    }

    fprintf(stdout, "PHASE0_END %lld\n", now_ms());
    fflush(stdout);

    cudaFree(d_data);
    free(h_data);
    return 0;
}