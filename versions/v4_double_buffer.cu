#include "../include/gemm_benchmark.cuh"

constexpr int TILE_SIZE = 16;
constexpr int THREAD_TILE_M = 4;
constexpr int THREAD_TILE_N = 4;
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 16;
constexpr int VECTOR_WIDTH = 4;
constexpr int BUFFER_COUNT = 2;

__device__ __forceinline__ float4 loadFloat4OrZero(
    const float* matrix,
    int row,
    int col,
    int rows,
    int cols)
{
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (row >= rows) return value;

    if (cols % VECTOR_WIDTH == 0 && col + VECTOR_WIDTH <= cols) {
        size_t offset = static_cast<size_t>(row) * cols + col;
        return *reinterpret_cast<const float4*>(matrix + offset);
    }

    if (col < cols) value.x = matrix[static_cast<size_t>(row) * cols + col];
    if (col + 1 < cols) value.y = matrix[static_cast<size_t>(row) * cols + col + 1];
    if (col + 2 < cols) value.z = matrix[static_cast<size_t>(row) * cols + col + 2];
    if (col + 3 < cols) value.w = matrix[static_cast<size_t>(row) * cols + col + 3];

    return value;
}

__global__ void gemmDoubleBuffered(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[BUFFER_COUNT][BLOCK_TILE_M][BLOCK_TILE_K + 1];
    __shared__ float tileB[BUFFER_COUNT][BLOCK_TILE_K][BLOCK_TILE_N];

    int blockRow = blockIdx.y * BLOCK_TILE_M;
    int blockCol = blockIdx.x * BLOCK_TILE_N;
    int linearThread = threadIdx.y * TILE_SIZE + threadIdx.x;

    int aVectorRow = linearThread / (BLOCK_TILE_K / VECTOR_WIDTH);
    int aVectorCol =
        (linearThread % (BLOCK_TILE_K / VECTOR_WIDTH)) * VECTOR_WIDTH;
    int bVectorRow = linearThread / (BLOCK_TILE_N / VECTOR_WIDTH);
    int bVectorCol =
        (linearThread % (BLOCK_TILE_N / VECTOR_WIDTH)) * VECTOR_WIDTH;

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {};

    float4 firstA = loadFloat4OrZero(
        A, blockRow + aVectorRow, aVectorCol, M, K
    );
    float4 firstB = loadFloat4OrZero(
        B, bVectorRow, blockCol + bVectorCol, K, N
    );

    tileA[0][aVectorRow][aVectorCol] = firstA.x;
    tileA[0][aVectorRow][aVectorCol + 1] = firstA.y;
    tileA[0][aVectorRow][aVectorCol + 2] = firstA.z;
    tileA[0][aVectorRow][aVectorCol + 3] = firstA.w;

    tileB[0][bVectorRow][bVectorCol] = firstB.x;
    tileB[0][bVectorRow][bVectorCol + 1] = firstB.y;
    tileB[0][bVectorRow][bVectorCol + 2] = firstB.z;
    tileB[0][bVectorRow][bVectorCol + 3] = firstB.w;

    __syncthreads();

    int currentBuffer = 0;

    for (int tileK = 0; tileK < K; tileK += BLOCK_TILE_K) {
        int nextTileK = tileK + BLOCK_TILE_K;
        int nextBuffer = currentBuffer ^ 1;
        bool hasNextTile = nextTileK < K;

        float4 nextA = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 nextB = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (hasNextTile) {
            nextA = loadFloat4OrZero(
                A,
                blockRow + aVectorRow,
                nextTileK + aVectorCol,
                M,
                K
            );

            nextB = loadFloat4OrZero(
                B,
                nextTileK + bVectorRow,
                blockCol + bVectorCol,
                K,
                N
            );
        }

        #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float regA[THREAD_TILE_M];
            float regB[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                regA[i] = tileA[currentBuffer][i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                regB[j] = tileB[currentBuffer][k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        if (hasNextTile) {
            tileA[nextBuffer][aVectorRow][aVectorCol] = nextA.x;
            tileA[nextBuffer][aVectorRow][aVectorCol + 1] = nextA.y;
            tileA[nextBuffer][aVectorRow][aVectorCol + 2] = nextA.z;
            tileA[nextBuffer][aVectorRow][aVectorCol + 3] = nextA.w;

            tileB[nextBuffer][bVectorRow][bVectorCol] = nextB.x;
            tileB[nextBuffer][bVectorRow][bVectorCol + 1] = nextB.y;
            tileB[nextBuffer][bVectorRow][bVectorCol + 2] = nextB.z;
            tileB[nextBuffer][bVectorRow][bVectorCol + 3] = nextB.w;

            __syncthreads();
            currentBuffer = nextBuffer;
        }
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

        gemmDoubleBuffered<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V4 Double Buffered GEMM",
        "Block tile: 64 x 64 x 16\nBuffers: 2\n",
        [] (float value) { return value; },
        launch
    );
}
