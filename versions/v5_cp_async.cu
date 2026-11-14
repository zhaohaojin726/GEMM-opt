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

__device__ __forceinline__ void copyFloat4GlobalToSharedAsync(
    float* destination,
    const float* matrix,
    int row,
    int col,
    int rows,
    int cols)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    bool canVectorize =
        row < rows
        && cols % VECTOR_WIDTH == 0
        && col % VECTOR_WIDTH == 0
        && col + VECTOR_WIDTH <= cols;

    if (canVectorize) {
        size_t offset = static_cast<size_t>(row) * cols + col;
        unsigned int sharedAddress = static_cast<unsigned int>(
            __cvta_generic_to_shared(destination)
        );

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(sharedAddress), "l"(matrix + offset)
        );

        return;
    }
#endif

    float4 value = loadFloat4OrZero(matrix, row, col, rows, cols);
    destination[0] = value.x;
    destination[1] = value.y;
    destination[2] = value.z;
    destination[3] = value.w;
}

__device__ __forceinline__ void commitAsyncCopies()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : :);
#endif
}

__device__ __forceinline__ void waitForAsyncCopies()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}

__global__ void gemmCpAsyncDoubleBuffered(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ __align__(16) float tileA
        [BUFFER_COUNT][BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ __align__(16) float tileB
        [BUFFER_COUNT][BLOCK_TILE_K][BLOCK_TILE_N];

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

    copyFloat4GlobalToSharedAsync(
        &tileA[0][aVectorRow][aVectorCol],
        A,
        blockRow + aVectorRow,
        aVectorCol,
        M,
        K
    );

    copyFloat4GlobalToSharedAsync(
        &tileB[0][bVectorRow][bVectorCol],
        B,
        bVectorRow,
        blockCol + bVectorCol,
        K,
        N
    );

    commitAsyncCopies();
    waitForAsyncCopies();
    __syncthreads();

    int currentBuffer = 0;

    for (int tileK = 0; tileK < K; tileK += BLOCK_TILE_K) {
        int nextTileK = tileK + BLOCK_TILE_K;
        int nextBuffer = currentBuffer ^ 1;
        bool hasNextTile = nextTileK < K;

        if (hasNextTile) {
            copyFloat4GlobalToSharedAsync(
                &tileA[nextBuffer][aVectorRow][aVectorCol],
                A,
                blockRow + aVectorRow,
                nextTileK + aVectorCol,
                M,
                K
            );

            copyFloat4GlobalToSharedAsync(
                &tileB[nextBuffer][bVectorRow][bVectorCol],
                B,
                nextTileK + bVectorRow,
                blockCol + bVectorCol,
                K,
                N
            );

            commitAsyncCopies();
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
            waitForAsyncCopies();
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
    int device = 0;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, gemmCpAsyncDoubleBuffered));

    if (properties.major < 8 || attributes.binaryVersion < 80) {
        std::cout
            << "V5 cp.async Double Buffered GEMM\n"
            << "Status: SKIPPED (requires sm_80+ device and build)\n";
        return EXIT_SUCCESS;
    }

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

        gemmCpAsyncDoubleBuffered<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<float>(
        "V5 cp.async Double Buffered GEMM",
        "Block tile: 64 x 64 x 16\nBuffers: 2\nCopy: cp.async 16 bytes\n",
        [] (float value) { return value; },
        launch
    );
}
