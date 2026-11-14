// Standalone Tensor Core GEMM demo: FP16 inputs, FP32 accumulation/output.
// Build (RTX 4060): nvcc -O3 -lineinfo -arch=sm_89 gemm_ncu.cu -o gemm_ncu
// Check + benchmark: ./gemm_ncu 512
// Profile exactly one launch, after 5 warmup launches:
// ncu --set full --kernel-name regex:gemm_tc --launch-skip 5 --launch-count 1 -f -o gemm_report ./gemm_ncu 1024 --profile
// Open: ncu-ui gemm_report.ncu-rep

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

using namespace nvcuda;
constexpr int BM = 64, BN = 64, BK = 16;

// Requires block=(32,4), M/N multiples of 64, K multiple of 16.
// Each of 4 warps owns a 32x32 output tile, using four 16x16 fragments.
__global__ void gemm_tc(const half* A, const half* B, float* C,
                        int M, int N, int K) {
    __shared__ __align__(32) half As[BM][BK];
    __shared__ __align__(32) half Bs[BK][BN];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * BM, col = blockIdx.x * BN;
    const int warp_r = ty / 2, warp_c = ty % 2;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int base_k = 0; base_k < K; base_k += BK) {
        // All threads cooperate to load one block tile.
        for (int i = ty; i < BM; i += 4)
            for (int j = tx; j < BK; j += 32)
                As[i][j] = A[static_cast<size_t>(row + i) * K + base_k + j];
        for (int i = ty; i < BK; i += 4)
            for (int j = tx; j < BN; j += 32)
                Bs[i][j] = B[static_cast<size_t>(base_k + i) * N + col + j];
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            wmma::load_matrix_sync(a[i], &As[warp_r * 32 + i * 16][0], BK);
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::load_matrix_sync(b[j], &Bs[0][warp_c * 32 + j * 16], BN);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        // Prevent overwriting shared memory while another warp still reads it.
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(
                C + static_cast<size_t>(row + warp_r * 32 + i * 16) * N
                  + col + warp_c * 32 + j * 16,
                acc[i][j], N, wmma::mem_row_major);
}

int main(int argc, char** argv) {
    int n = 512;
    bool profile = false;
    try {
        if (argc > 3) throw std::invalid_argument("arguments");
        if (argc >= 2) {
            size_t used = 0;
            n = std::stoi(argv[1], &used);
            if (used != std::string(argv[1]).size()) throw std::invalid_argument("size");
        }
        if (argc == 3) {
            if (std::string(argv[2]) != "--profile") throw std::invalid_argument("mode");
            profile = true;
        }
        if (n <= 0 || n % 64 != 0) throw std::invalid_argument("size");
    } catch (const std::exception&) {
        std::fprintf(stderr, "Usage: %s [N: positive multiple of 64] [--profile]\n", argv[0]);
        return EXIT_FAILURE;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 7) {
        std::fprintf(stderr, "This example requires Tensor Core support (SM 70+).\n");
        return EXIT_FAILURE;
    }
    const size_t count = static_cast<size_t>(n) * n;
    std::vector<half> A(count), B(count);
    std::vector<float> C(count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-4, 4);
    // Exactly representable FP16 values simplify correctness checking.
    for (size_t i = 0; i < count; ++i) {
        A[i] = __float2half(dist(rng) / 8.0f);
        B[i] = __float2half(dist(rng) / 8.0f);
    }
    half *dA = nullptr, *dB = nullptr;
    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dA), count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dB), count * sizeof(half)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dC), count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), count * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), count * sizeof(half), cudaMemcpyHostToDevice));
    auto launch = [&]() {
        gemm_tc<<<dim3(n / BN, n / BM), dim3(32, 4)>>>(dA, dB, dC, n, n, n);
        CUDA_CHECK(cudaGetLastError());
    };
    // Exactly five matching kernel launches precede the measured launch(s).
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    const int iterations = profile ? 1 : 50;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(C.data(), dC, count * sizeof(float), cudaMemcpyDeviceToHost));

    // Full CPU verification for N<=256; otherwise verify corners + 256 samples.
    const bool full_check = n <= 256;
    const size_t checks = full_check ? count : 260;
    std::uniform_int_distribution<int> index(0, n - 1);
    double max_error = 0;
    bool ok = true;
    for (size_t p = 0; p < checks; ++p) {
        int r, c;
        if (full_check) { r = static_cast<int>(p / n); c = static_cast<int>(p % n); }
        else if (p < 4) { r = (p / 2) * (n - 1); c = (p % 2) * (n - 1); }
        else { r = index(rng); c = index(rng); }
        double ref = 0;
        for (int k = 0; k < n; ++k)
            ref += static_cast<double>(__half2float(A[static_cast<size_t>(r) * n + k]))
                 * __half2float(B[static_cast<size_t>(k) * n + c]);
        const float result = C[static_cast<size_t>(r) * n + c];
        const double error = std::abs(result - ref);
        max_error = std::max(max_error, error);
        if (!std::isfinite(result) || error > 1e-3 + 1e-4 * std::abs(ref)) ok = false;
    }
    std::printf("GPU: %s\nGEMM: %d x %d x %d, block=(32,4), tile=(64,64,16)\n",
                prop.name, n, n, n);
    std::printf("CPU check: %s (%s, %zu outputs), max abs error: %.8g\n",
                ok ? "PASS" : "FAIL", full_check ? "full" : "sampled", checks, max_error);
    if (profile) std::puts("Profile mode: use kernel duration from NCU; event timing is distorted by profiling.");
    else {
        const double avg_ms = ms / iterations;
        std::printf("Average: %.6f ms; %.3f TFLOP/s\n", avg_ms,
                    2.0 * n * n * n / (avg_ms * 1e9));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
