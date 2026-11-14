#include "../include/gemm_benchmark.cuh"

__global__ void gemmNaive(
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

int main()
{
    auto launch = [](
        const float* A,
        const float* B,
        float* C,
        int M,
        int N,
        int K) {
        dim3 block(16, 16);
        dim3 grid(
            (N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y
        );

        gemmNaive<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V0 Naive GEMM",
        "Thread output: 1 x 1\n",
        [] (float value) { return value; },
        launch
    );
}
