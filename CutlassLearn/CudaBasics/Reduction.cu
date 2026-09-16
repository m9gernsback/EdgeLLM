// Stage 0 / Exercise 2: Reduction (sum of N floats)
//
// Goal: learn block-level cooperation — the thing compute shaders rarely
// force you to think about explicitly.
//
// V1: classic shared-memory tree reduction (like groupshared + barriers),
//     one atomicAdd per block to combine partial sums.
// V2: warp-shuffle reduction — threads in a warp exchange values via
//     __shfl_down_sync registers instead of shared memory, so the final
//     32-element reduction needs no __syncthreads at all.
//
// Note: float addition is not associative, so GPU and CPU results differ
// slightly. We check with a loose tolerance instead of exact equality.

#include <cstdio>
#include <cuda_runtime.h>

#include "CudaUtils.cuh"

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

// V1: shared-memory tree reduction.
__global__ void ReduceShared(const float* __restrict__ in,
                             float* __restrict__ out, int n) {
  __shared__ float smem[kBlockSize];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  smem[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, smem[0]);
}

// V2: intra-warp shuffle + one shared-memory round across warps.
__global__ void ReduceShuffle(const float* __restrict__ in,
                              float* __restrict__ out, int n) {
  __shared__ float warp_sums[kBlockSize / kWarpSize];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float sum = (i < n) ? in[i] : 0.0f;

  // Reduce within the warp using register exchanges (no barriers).
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);

  int lane = tid & (kWarpSize - 1);
  int warp = tid / kWarpSize;
  if (lane == 0) warp_sums[warp] = sum;
  __syncthreads();

  // First warp reduces the per-warp partials.
  if (warp == 0) {
    sum = (lane < kBlockSize / kWarpSize) ? warp_sums[lane] : 0.0f;
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) atomicAdd(out, sum);
  }
}

int main() {
  const int n = 1 << 26;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  float* h_in = new float[n];
  FillRandom(h_in, n, 3);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  const int grid = (n + kBlockSize - 1) / kBlockSize;
  float gpu_sum = 0.f;

  auto run = [&](auto kernel, const char* name) {
    auto launch = [&] {
      CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
      kernel<<<grid, kBlockSize>>>(d_in, d_out, n);
    };
    launch();
    CUDA_CHECK(cudaGetLastError());
    float ms = BenchmarkMs(launch);
    printf("%-24s %8.3f ms   %7.1f GB/s\n", name, ms, bytes / ms / 1e6);

    // Grab the result from the last benchmark iteration.
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    kernel<<<grid, kBlockSize>>>(d_in, d_out, n);
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return gpu_sum;
  };

  float sum_v1 = run(ReduceShared, "V1 shared-mem tree");
  float sum_v2 = run(ReduceShuffle, "V2 warp shuffle");

  // CPU reference (double to keep the reference accurate).
  double ref = 0.0;
  for (int i = 0; i < n; ++i) ref += h_in[i];

  // Relative tolerance: reduction order changes float rounding, and the
  // absolute gap grows with the magnitude of the sum.
  auto ok = [&](float s) {
    return fabsf(s - (float)ref) < 1e-4f * (fabsf((float)ref) + 1.0f);
  };
  printf("CPU ref %.4f | V1 %.4f %s | V2 %.4f %s\n", ref, sum_v1,
         ok(sum_v1) ? "PASS" : "FAIL", sum_v2, ok(sum_v2) ? "PASS" : "FAIL");

  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  return (ok(sum_v1) && ok(sum_v2)) ? 0 : 1;
}
