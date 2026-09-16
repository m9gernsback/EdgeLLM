// Stage 0 / Exercise 1: VectorAdd
//
// Goal: map compute-shader mental model onto CUDA and measure memory
// bandwidth. This kernel is pure memory-bound: 2 reads + 1 write per
// element, so achieved GB/s is the only metric that matters.
//
// V1: one thread per element (like a compute shader with numthreads(256,1,1)
//     and exactly enough groups).
// V2: grid-stride loop — a fixed-size grid where each thread processes
//     multiple elements. This is the idiomatic CUDA pattern; compare its
//     bandwidth against V1 and think about why they differ (or don't).

#include <cstdio>
#include <cuda_runtime.h>

#include "CudaUtils.cuh"

constexpr int kBlockSize = 256;

__global__ void VectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b, float* __restrict__ c,
                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void VectorAddGridStride(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}

static void ReportBandwidth(const char* name, float ms, size_t bytes) {
  printf("%-24s %8.3f ms   %7.1f GB/s\n", name, ms, bytes / ms / 1e6);
}

int main() {
  const int n = 1 << 26;  // 64M elements; 3 arrays * 256 MB
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float *h_a = new float[n], *h_b = new float[n], *h_c = new float[n];
  FillRandom(h_a, n, 1);
  FillRandom(h_b, n, 2);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  const size_t traffic = 3 * bytes;  // read a, read b, write c

  // --- V1: one thread per element ---
  {
    int grid = (n + kBlockSize - 1) / kBlockSize;
    auto launch = [&] {
      VectorAdd<<<grid, kBlockSize>>>(d_a, d_b, d_c, n);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    float ms = BenchmarkMs(launch);
    ReportBandwidth("V1 one-thread-per-elem", ms, traffic);
  }

  // --- V2: grid-stride loop, fixed grid sized to fill the device ---
  {
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count,
                                      cudaDevAttrMultiProcessorCount, 0));
    int grid = sm_count * 8;  // 8 blocks per SM is plenty for this kernel
    auto launch = [&] {
      VectorAddGridStride<<<grid, kBlockSize>>>(d_a, d_b, d_c, n);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    float ms = BenchmarkMs(launch);
    ReportBandwidth("V2 grid-stride", ms, traffic);
  }

  // --- correctness ---
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) h_a[i] += h_b[i];  // CPU reference in-place
  int bad = CheckClose(h_c, h_a, n);
  if (bad >= 0) {
    printf("FAIL at index %d: got %f, want %f\n", bad, h_c[bad], h_a[bad]);
    return 1;
  }
  printf("PASS (n = %d)\n", n);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  return 0;
}
