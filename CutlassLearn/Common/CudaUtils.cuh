// Common utilities for all CutlassLearn exercises:
// error checking, GPU timing, benchmark harness, correctness checks.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
              __LINE__, cudaGetErrorString(err_));                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// RAII wrapper around cudaEvent for kernel timing.
struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start() { CUDA_CHECK(cudaEventRecord(start)); }
  // Returns elapsed milliseconds since Start().
  float Stop() {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// Run `fn` warmup times (untimed), then `iters` times; return average ms.
template <typename Fn>
float BenchmarkMs(Fn&& fn, int warmup = 3, int iters = 20) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  GpuTimer timer;
  timer.Start();
  for (int i = 0; i < iters; ++i) fn();
  return timer.Stop() / iters;
}

inline void FillRandom(float* data, int n, unsigned seed = 42) {
  srand(seed);
  for (int i = 0; i < n; ++i)
    data[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
}

// Element-wise check with relative+absolute tolerance. Returns first bad
// index or -1 on success.
inline int CheckClose(const float* got, const float* want, int n,
                      float rtol = 1e-4f, float atol = 1e-5f) {
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(got[i] - want[i]);
    float tol = atol + rtol * fabsf(want[i]);
    if (diff > tol) return i;
  }
  return -1;
}
