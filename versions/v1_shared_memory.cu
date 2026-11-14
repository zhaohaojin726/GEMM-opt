#include "../include/gemm_benchmark.cuh"

constexpr int TILE_SIZE = 16;

__global__ void gemmSharedMemory(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    float sum = 0.0f;

    int tileCount = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tileCount; ++tile) {
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K)
                ? A[static_cast<size_t>(row) * K + aCol]
                : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N)
                ? B[static_cast<size_t>(bRow) * N + col]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k]
                 * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = sum;
    }
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
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(
            (N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE
        );

        gemmSharedMemory<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V1 Shared Memory Tiling GEMM",
        "Block tile: 16 x 16 x 16\n",
        [] (float value) { return value; },
        launch
    );
}
