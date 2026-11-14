#include "../include/gemm_benchmark.cuh"

constexpr int WARP_SIZE_VALUE = 32;
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 16;
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;
constexpr int WARPS_M = BLOCK_TILE_M / WARP_TILE_M;
constexpr int WARPS_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE_VALUE;
constexpr int WARP_LANES_M = 4;
constexpr int WARP_LANES_N = 8;
constexpr int THREAD_TILE_M = WARP_TILE_M / WARP_LANES_M;
constexpr int THREAD_TILE_N = WARP_TILE_N / WARP_LANES_N;
constexpr int VECTOR_WIDTH = 4;

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

__global__ void gemmWarpTiled(
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
    int linearThread = threadIdx.y * WARP_SIZE_VALUE + threadIdx.x;
    int warpId = linearThread / WARP_SIZE_VALUE;
    int laneId = linearThread % WARP_SIZE_VALUE;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int laneRow = laneId / WARP_LANES_N;
    int laneCol = laneId % WARP_LANES_N;

    float accum[THREAD_TILE_M][THREAD_TILE_N] = {};

    constexpr int aVectorsPerTile =
        BLOCK_TILE_M * BLOCK_TILE_K / VECTOR_WIDTH;
    constexpr int bVectorsPerTile =
        BLOCK_TILE_K * BLOCK_TILE_N / VECTOR_WIDTH;
    constexpr int aVectorsPerRow = BLOCK_TILE_K / VECTOR_WIDTH;
    constexpr int bVectorsPerRow = BLOCK_TILE_N / VECTOR_WIDTH;

    for (int tileK = 0; tileK < K; tileK += BLOCK_TILE_K) {
        #pragma unroll
        for (int vectorIndex = linearThread;
             vectorIndex < aVectorsPerTile;
             vectorIndex += BLOCK_THREADS) {
            int tileRow = vectorIndex / aVectorsPerRow;
            int tileCol =
                (vectorIndex % aVectorsPerRow) * VECTOR_WIDTH;

            float4 value = loadFloat4OrZero(
                A,
                blockRow + tileRow,
                tileK + tileCol,
                M,
                K
            );

            tileA[tileRow][tileCol] = value.x;
            tileA[tileRow][tileCol + 1] = value.y;
            tileA[tileRow][tileCol + 2] = value.z;
            tileA[tileRow][tileCol + 3] = value.w;
        }

        #pragma unroll
        for (int vectorIndex = linearThread;
             vectorIndex < bVectorsPerTile;
             vectorIndex += BLOCK_THREADS) {
            int tileRow = vectorIndex / bVectorsPerRow;
            int tileCol =
                (vectorIndex % bVectorsPerRow) * VECTOR_WIDTH;

            float4 value = loadFloat4OrZero(
                B,
                tileK + tileRow,
                blockCol + tileCol,
                K,
                N
            );

            tileB[tileRow][tileCol] = value.x;
            tileB[tileRow][tileCol + 1] = value.y;
            tileB[tileRow][tileCol + 2] = value.z;
            tileB[tileRow][tileCol + 3] = value.w;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float regA[THREAD_TILE_M];
            float regB[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                int rowInTile =
                    warpRow * WARP_TILE_M
                    + i * WARP_LANES_M
                    + laneRow;

                regA[i] = tileA[rowInTile][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int colInTile =
                    warpCol * WARP_TILE_N
                    + j * WARP_LANES_N
                    + laneCol;

                regB[j] = tileB[k][colInTile];
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
        int row =
            blockRow
            + warpRow * WARP_TILE_M
            + i * WARP_LANES_M
            + laneRow;

        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            int col =
                blockCol
                + warpCol * WARP_TILE_N
                + j * WARP_LANES_N
                + laneCol;

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
        dim3 block(WARP_SIZE_VALUE, WARPS_PER_BLOCK);
        dim3 grid(
            (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
            (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M
        );

        gemmWarpTiled<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V6 Warp Tiling GEMM",
        "Block tile: 64 x 64 x 16\nWarp tile: 32 x 32\nThread tile: 8 x 4\n",
        [] (float value) { return value; },
        launch
    );
}
