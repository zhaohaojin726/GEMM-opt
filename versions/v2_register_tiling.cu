#include "../include/gemm_benchmark.cuh"

constexpr int TILE_SIZE = 16;
constexpr int THREAD_TILE_M = 4;
constexpr int THREAD_TILE_N = 4;
constexpr int BLOCK_TILE_M = TILE_SIZE * THREAD_TILE_M;
constexpr int BLOCK_TILE_N = TILE_SIZE * THREAD_TILE_N;
constexpr int BLOCK_TILE_K = TILE_SIZE;
constexpr int THREAD_COUNT = TILE_SIZE * TILE_SIZE;

__global__ void gemmRegisterTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[BLOCK_TILE_M][BLOCK_TILE_K + 1];
    __shared__ float tileB[BLOCK_TILE_K][BLOCK_TILE_N];

    int blockRow = blockIdx.y * BLOCK_TILE_M;
    int blockCol = blockIdx.x * BLOCK_TILE_N;
    int linearThread = threadIdx.y * TILE_SIZE + threadIdx.x;

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {};

    for (int tileK = 0; tileK < K; tileK += BLOCK_TILE_K) {
        for (int index = linearThread;
             index < BLOCK_TILE_M * BLOCK_TILE_K;
             index += THREAD_COUNT) {
            int tileRow = index / BLOCK_TILE_K;
            int tileCol = index % BLOCK_TILE_K;
            int row = blockRow + tileRow;
            int col = tileK + tileCol;

            tileA[tileRow][tileCol] =
                (row < M && col < K)
                    ? A[static_cast<size_t>(row) * K + col]
                    : 0.0f;
        }

        for (int index = linearThread;
             index < BLOCK_TILE_K * BLOCK_TILE_N;
             index += THREAD_COUNT) {
            int tileRow = index / BLOCK_TILE_N;
            int tileCol = index % BLOCK_TILE_N;
            int row = tileK + tileRow;
            int col = blockCol + tileCol;

            tileB[tileRow][tileCol] =
                (row < K && col < N)
                    ? B[static_cast<size_t>(row) * N + col]
                    : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float regA[THREAD_TILE_M];
            float regB[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                regA[i] = tileA[i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                regB[j] = tileB[k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int row = blockRow + i * TILE_SIZE + threadIdx.y;

        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int col = blockCol + j * TILE_SIZE + threadIdx.x;

            if (row < M && col < N) {
                C[static_cast<size_t>(row) * N + col] = accum[i][j];
            }
        }
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
            (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
            (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M
        );

        gemmRegisterTiled<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V2 Register Tiling GEMM",
        "Block tile: 64 x 64 x 16\nThread tile: 4 x 4\n",
        [] (float value) { return value; },
        launch
    );
}
