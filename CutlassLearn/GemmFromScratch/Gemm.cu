// Stage 1: GEMM from scratch — C = A * B, all row-major FP32.
//   A: M x K, B: K x N, C: M x N
//
// This file is the "optimization ladder": each Vx kernel adds one classic
// optimization on top of the previous one. All versions are checked against
// cuBLAS and reported in TFLOPS.
//
//   V1 naive          — one thread per output element, all reads from global
//   V2 smem tiling    — (next exercise) stage tiles in shared memory
//   V3 register tiling
//   V4 vectorized loads + double buffering
//   V5 swizzled smem  — bank-conflict-free
//
// Reference: Simon Boehm, "How to Optimize a CUDA Matmul Kernel for
// cuBLAS-like Performance" — work through it alongside these kernels.

#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "CudaUtils.cuh"

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,        \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// V1: naive — one thread computes one C element.
//
// Every thread walks a full row of A and a full column of B straight from
// global memory. A warp reads B coalesced (consecutive col), and A is the
// same address across the warp (broadcast), so it's not a total disaster —
// the real problem is *zero data reuse*: each element of A is re-fetched
// from global memory N times, each element of B is re-fetched M times.
// Global traffic = 2*K floats per output element.
// ---------------------------------------------------------------------------
__global__ void GemmNaive(const float* __restrict__ A,
                          const float* __restrict__ B, float* __restrict__ C,
                          int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct GemmVersion {
  const char* name;
  void (*launch)(const float*, const float*, float*, int, int, int);
  int warmup, iters;  // slow versions get fewer iterations
};

static void LaunchNaive(const float* A, const float* B, float* C, int M,
                        int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  GemmNaive<<<grid, block>>>(A, B, C, M, N, K);
}

int main() {
  const int M = 4096, N = 4096, K = 4096;
  const size_t bytes_a = (size_t)M * K * sizeof(float);
  const size_t bytes_b = (size_t)K * N * sizeof(float);
  const size_t bytes_c = (size_t)M * N * sizeof(float);
  const double flops = 2.0 * M * N * K;

  float* h_a = new float[(size_t)M * K];
  float* h_b = new float[(size_t)K * N];
  FillRandom(h_a, M * K, 1);
  FillRandom(h_b, K * N, 2);

  float *d_a, *d_b, *d_c, *d_ref;
  CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
  CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
  CUDA_CHECK(cudaMalloc(&d_c, bytes_c));
  CUDA_CHECK(cudaMalloc(&d_ref, bytes_c));
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));
  delete[] h_a;
  delete[] h_b;

  // --- cuBLAS reference (also our perf target) ---
  // cuBLAS is column-major. A row-major MxK matrix is a column-major KxM
  // matrix, so C_rowmajor = A*B is computed as C_colmajor' = B' * A':
  // pass (B, A) with swapped roles and dimensions (N, M, K).
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;
  auto cublas_run = [&](float* out) {
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_b, N, d_a, K, &beta, out, N));
  };
  cublas_run(d_ref);

  float* h_ref = new float[(size_t)M * N];
  CUDA_CHECK(cudaMemcpy(h_ref, d_ref, bytes_c, cudaMemcpyDeviceToHost));

  float ms_ref = BenchmarkMs([&] { cublas_run(d_ref); }, 3, 10);
  printf("%-12s %9.3f ms   %7.2f TFLOPS   (reference)\n", "cuBLAS", ms_ref,
         flops / ms_ref / 1e9);

  // --- optimization ladder ---
  std::vector<GemmVersion> versions = {
      {"V1 naive", LaunchNaive, 1, 3},
      // V2, V3, ... go here.
  };

  float* h_c = new float[(size_t)M * N];
  bool all_pass = true;
  for (auto& v : versions) {
    v.launch(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));
    // FP32 accumulation order differs; GEMM over K=4096 needs a loose rtol.
    int bad = CheckClose(h_c, h_ref, M * N, /*rtol=*/1e-3f, /*atol=*/1e-3f);
    bool pass = (bad < 0);
    all_pass &= pass;
    if (!pass)
      printf("  first mismatch at %d: got %f, want %f\n", bad, h_c[bad],
             h_ref[bad]);

    float ms = BenchmarkMs([&] { v.launch(d_a, d_b, d_c, M, N, K); }, v.warmup,
                           v.iters);
    printf("%-12s %9.3f ms   %7.2f TFLOPS   %5.1f%% of cuBLAS   %s\n", v.name,
           ms, flops / ms / 1e9, 100.0 * ms_ref / ms,
           pass ? "PASS" : "FAIL");
  }

  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_ref);
  delete[] h_c;
  delete[] h_ref;
  return all_pass ? 0 : 1;
}
