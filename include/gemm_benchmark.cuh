#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

namespace gemm_demo {

inline constexpr int MATRIX_M = 1024;
inline constexpr int MATRIX_N = 1024;
inline constexpr int MATRIX_K = 1024;
inline constexpr int WARMUP_COUNT = 5;
inline constexpr int REPEAT_COUNT = 20;

// 独立版本统一使用这个普通 FP32 kernel 生成正确性参考结果。
static __global__ void referenceGemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[static_cast<size_t>(row) * K + k]
             * B[static_cast<size_t>(k) * N + col];
    }

    C[static_cast<size_t>(row) * N + col] = sum;
}

inline void initializeInputs(
    std::vector<float>& A,
    std::vector<float>& B)
{
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.1f;
    }

    for (size_t i = 0; i < B.size(); ++i) {
        B[i] =
            static_cast<float>(
                static_cast<int>(i % 5) - 2
            ) * 0.1f;
    }
}

template <typename Launch>
float benchmarkKernel(Launch launch)
{
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT_COUNT; ++i) {
        launch();
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / REPEAT_COUNT;
}

template <typename InputType, typename Convert, typename Launch>
int runVersion(
    const char* versionName,
    const char* configuration,
    Convert convert,
    Launch launch,
    float tolerance = 1e-3f)
{
    constexpr int M = MATRIX_M;
    constexpr int N = MATRIX_N;
    constexpr int K = MATRIX_K;

    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<float> hAFloat(sizeA);
    std::vector<float> hBFloat(sizeB);
    std::vector<InputType> hA(sizeA);
    std::vector<InputType> hB(sizeB);
    std::vector<float> hC(sizeC);
    std::vector<float> hReference(sizeC);

    initializeInputs(hAFloat, hBFloat);

    for (size_t i = 0; i < sizeA; ++i) {
        hA[i] = convert(hAFloat[i]);
    }

    for (size_t i = 0; i < sizeB; ++i) {
        hB[i] = convert(hBFloat[i]);
    }

    InputType* dA = nullptr;
    InputType* dB = nullptr;
    float* dC = nullptr;
    float* dReferenceA = nullptr;
    float* dReferenceB = nullptr;
    float* dReferenceC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(InputType)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(InputType)));
    CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dReferenceA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dReferenceB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dReferenceC, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        dA,
        hA.data(),
        sizeA * sizeof(InputType),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dB,
        hB.data(),
        sizeB * sizeof(InputType),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dReferenceA,
        hAFloat.data(),
        sizeA * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        dReferenceB,
        hBFloat.data(),
        sizeB * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 referenceBlock(16, 16);
    dim3 referenceGrid(
        (N + referenceBlock.x - 1) / referenceBlock.x,
        (M + referenceBlock.y - 1) / referenceBlock.y
    );

    referenceGemm<<<referenceGrid, referenceBlock>>>(
        dReferenceA,
        dReferenceB,
        dReferenceC,
        M,
        N,
        K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedMs = [&]() {
        auto launchCurrentVersion = [&]() {
            launch(dA, dB, dC, M, N, K);
        };

        return benchmarkKernel(launchCurrentVersion);
    }();

    CUDA_CHECK(cudaMemcpy(
        hC.data(),
        dC,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        hReference.data(),
        dReferenceC,
        sizeC * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float maxAbsError = 0.0f;

    for (size_t i = 0; i < sizeC; ++i) {
        maxAbsError = std::fmax(
            maxAbsError,
            std::fabs(hReference[i] - hC[i])
        );
    }

    double flops = 2.0 * M * N * K;
    double gflops = flops / (elapsedMs / 1000.0) / 1e9;
    bool passed = maxAbsError < tolerance;

    std::cout
        << std::fixed
        << std::setprecision(3)
        << "=================================\n"
        << versionName << "\n"
        << "Matrix: " << M << " x " << N << " x " << K << "\n"
        << configuration
        << "Time: " << elapsedMs << " ms\n"
        << "Performance: " << gflops << " GFLOPS\n"
        << std::setprecision(6)
        << "Max absolute error: " << maxAbsError << "\n"
        << "Tolerance: " << tolerance << "\n"
        << "Check: " << (passed ? "PASSED" : "FAILED") << "\n"
        << "=================================\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dReferenceA));
    CUDA_CHECK(cudaFree(dReferenceB));
    CUDA_CHECK(cudaFree(dReferenceC));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace gemm_demo
