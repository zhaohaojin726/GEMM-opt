#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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


// ============================================================
// V0: Naive GEMM
//
// 一个线程计算一个 C[row][col]
// ============================================================

__global__ void gemmNaive(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k]
             * B[k * N + col];
    }

    C[row * N + col] = sum;
}


// ============================================================
// V1: Shared Memory Tiling GEMM
//
// 一个 block 计算 C 的一个 TILE_SIZE x TILE_SIZE 分块。
// A、B 的分块先由整个 block 协作加载到 shared memory，再复用。
// ============================================================

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

    int col =
        blockIdx.x * TILE_SIZE
        + threadIdx.x;

    int row =
        blockIdx.y * TILE_SIZE
        + threadIdx.y;

    float sum = 0.0f;

    // 沿 K 维逐块计算。最后一块不足 TILE_SIZE 的位置补 0，
    // 因而 M、N、K 都不要求是 TILE_SIZE 的整数倍。
    int numTiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int aCol =
            tile * TILE_SIZE
            + threadIdx.x;

        int bRow =
            tile * TILE_SIZE
            + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K)
                ? A[row * K + aCol]
                : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N)
                ? B[bRow * N + col]
                : 0.0f;

        // 确保当前 tile 已经全部加载完成。
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k]
                 * tileB[k][threadIdx.x];
        }

        // 确保所有线程使用完当前 tile 后再装载下一块。
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================
// V2: Shared Memory + Register Tiling GEMM
//
// 一个 16 x 16 block 计算 C 的一个 64 x 64 分块；每个线程
// 用 16 个寄存器累加一个 4 x 4 输出子块。
// ============================================================

constexpr int REGISTER_TILE_M = 4;
constexpr int REGISTER_TILE_N = 4;
constexpr int REGISTER_BLOCK_M = TILE_SIZE * REGISTER_TILE_M;
constexpr int REGISTER_BLOCK_N = TILE_SIZE * REGISTER_TILE_N;
constexpr int REGISTER_BLOCK_K = TILE_SIZE;
constexpr int REGISTER_THREADS = TILE_SIZE * TILE_SIZE;

__global__ void gemmRegisterTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // A 的第二维补 1，减少相邻行访问 shared memory 时的 bank conflict。
    __shared__ float tileA[REGISTER_BLOCK_M][REGISTER_BLOCK_K + 1];
    __shared__ float tileB[REGISTER_BLOCK_K][REGISTER_BLOCK_N];

    int blockRow =
        blockIdx.y * REGISTER_BLOCK_M;

    int blockCol =
        blockIdx.x * REGISTER_BLOCK_N;

    int linearThread =
        threadIdx.y * TILE_SIZE
        + threadIdx.x;

    float accum[REGISTER_TILE_M][REGISTER_TILE_N];

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    for (int tileK = 0; tileK < K; tileK += REGISTER_BLOCK_K) {
        // 256 个线程协作加载 64 x 16 的 A tile。
        #pragma unroll
        for (int index = linearThread;
             index < REGISTER_BLOCK_M * REGISTER_BLOCK_K;
             index += REGISTER_THREADS) {
            int tileRow = index / REGISTER_BLOCK_K;
            int tileCol = index % REGISTER_BLOCK_K;
            int globalRow = blockRow + tileRow;
            int globalCol = tileK + tileCol;

            tileA[tileRow][tileCol] =
                (globalRow < M && globalCol < K)
                    ? A[globalRow * K + globalCol]
                    : 0.0f;
        }

        // 256 个线程协作加载 16 x 64 的 B tile。
        #pragma unroll
        for (int index = linearThread;
             index < REGISTER_BLOCK_K * REGISTER_BLOCK_N;
             index += REGISTER_THREADS) {
            int tileRow = index / REGISTER_BLOCK_N;
            int tileCol = index % REGISTER_BLOCK_N;
            int globalRow = tileK + tileRow;
            int globalCol = blockCol + tileCol;

            tileB[tileRow][tileCol] =
                (globalRow < K && globalCol < N)
                    ? B[globalRow * N + globalCol]
                    : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < REGISTER_BLOCK_K; ++k) {
            float regA[REGISTER_TILE_M];
            float regB[REGISTER_TILE_N];

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                regA[i] =
                    tileA[i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < REGISTER_TILE_N; ++j) {
                regB[j] =
                    tileB[k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REGISTER_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        int row =
            blockRow
            + i * TILE_SIZE
            + threadIdx.y;

        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            int col =
                blockCol
                + j * TILE_SIZE
                + threadIdx.x;

            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


// ============================================================
// V3: Vectorized Load + Register Tiling GEMM
//
// 计算分块与 V2 相同，但 global memory -> shared memory 使用
// float4（128-bit）向量化加载，减少 load 指令数量。
// ============================================================

constexpr int VECTOR_WIDTH = 4;

__device__ __forceinline__ float4 loadFloat4OrZero(
    const float* matrix,
    int row,
    int col,
    int rows,
    int cols)
{
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (row >= rows) {
        return value;
    }

    // cudaMalloc 返回的地址满足对齐要求。只要行步长是 4 的倍数，
    // 且当前向量没有跨越行尾，就可以安全使用 128-bit load。
    if ((cols % VECTOR_WIDTH) == 0
        && col + VECTOR_WIDTH <= cols) {
        size_t offset =
            static_cast<size_t>(row) * cols
            + col;

        return *reinterpret_cast<const float4*>(matrix + offset);
    }

    // 非对齐矩阵或 K/N 尾块使用标量回退，避免越界和非对齐访问。
    if (col < cols) {
        value.x = matrix[static_cast<size_t>(row) * cols + col];
    }

    if (col + 1 < cols) {
        value.y = matrix[static_cast<size_t>(row) * cols + col + 1];
    }

    if (col + 2 < cols) {
        value.z = matrix[static_cast<size_t>(row) * cols + col + 2];
    }

    if (col + 3 < cols) {
        value.w = matrix[static_cast<size_t>(row) * cols + col + 3];
    }

    return value;
}

__global__ void gemmVectorizedRegisterTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA[REGISTER_BLOCK_M][REGISTER_BLOCK_K + 1];
    __shared__ float tileB[REGISTER_BLOCK_K][REGISTER_BLOCK_N];

    int blockRow =
        blockIdx.y * REGISTER_BLOCK_M;

    int blockCol =
        blockIdx.x * REGISTER_BLOCK_N;

    int linearThread =
        threadIdx.y * TILE_SIZE
        + threadIdx.x;

    float accum[REGISTER_TILE_M][REGISTER_TILE_N];

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    for (int tileK = 0; tileK < K; tileK += REGISTER_BLOCK_K) {
        // A tile 有 64 x 16 个 float，恰好由 256 个线程
        // 每线程加载一个 float4。
        int aVectorRow =
            linearThread
            / (REGISTER_BLOCK_K / VECTOR_WIDTH);

        int aVectorCol =
            (linearThread % (REGISTER_BLOCK_K / VECTOR_WIDTH))
            * VECTOR_WIDTH;

        float4 vectorA =
            loadFloat4OrZero(
                A,
                blockRow + aVectorRow,
                tileK + aVectorCol,
                M,
                K
            );

        tileA[aVectorRow][aVectorCol] = vectorA.x;
        tileA[aVectorRow][aVectorCol + 1] = vectorA.y;
        tileA[aVectorRow][aVectorCol + 2] = vectorA.z;
        tileA[aVectorRow][aVectorCol + 3] = vectorA.w;

        // B tile 有 16 x 64 个 float，同样每线程加载一个 float4。
        int bVectorRow =
            linearThread
            / (REGISTER_BLOCK_N / VECTOR_WIDTH);

        int bVectorCol =
            (linearThread % (REGISTER_BLOCK_N / VECTOR_WIDTH))
            * VECTOR_WIDTH;

        float4 vectorB =
            loadFloat4OrZero(
                B,
                tileK + bVectorRow,
                blockCol + bVectorCol,
                K,
                N
            );

        tileB[bVectorRow][bVectorCol] = vectorB.x;
        tileB[bVectorRow][bVectorCol + 1] = vectorB.y;
        tileB[bVectorRow][bVectorCol + 2] = vectorB.z;
        tileB[bVectorRow][bVectorCol + 3] = vectorB.w;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < REGISTER_BLOCK_K; ++k) {
            float regA[REGISTER_TILE_M];
            float regB[REGISTER_TILE_N];

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                regA[i] =
                    tileA[i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < REGISTER_TILE_N; ++j) {
                regB[j] =
                    tileB[k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REGISTER_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        int row =
            blockRow
            + i * TILE_SIZE
            + threadIdx.y;

        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            int col =
                blockCol
                + j * TILE_SIZE
                + threadIdx.x;

            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


// ============================================================
// V4: Double Buffered GEMM
//
// 两套 shared-memory tile 交替使用。计算当前 tile 时，下一块
// A/B 先通过 float4 预取到寄存器，隐藏一部分 global-load 延迟。
// ============================================================

constexpr int BUFFER_COUNT = 2;

__global__ void gemmDoubleBuffered(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA
        [BUFFER_COUNT][REGISTER_BLOCK_M][REGISTER_BLOCK_K + 1];

    __shared__ float tileB
        [BUFFER_COUNT][REGISTER_BLOCK_K][REGISTER_BLOCK_N];

    int blockRow =
        blockIdx.y * REGISTER_BLOCK_M;

    int blockCol =
        blockIdx.x * REGISTER_BLOCK_N;

    int linearThread =
        threadIdx.y * TILE_SIZE
        + threadIdx.x;

    int aVectorRow =
        linearThread
        / (REGISTER_BLOCK_K / VECTOR_WIDTH);

    int aVectorCol =
        (linearThread % (REGISTER_BLOCK_K / VECTOR_WIDTH))
        * VECTOR_WIDTH;

    int bVectorRow =
        linearThread
        / (REGISTER_BLOCK_N / VECTOR_WIDTH);

    int bVectorCol =
        (linearThread % (REGISTER_BLOCK_N / VECTOR_WIDTH))
        * VECTOR_WIDTH;

    float accum[REGISTER_TILE_M][REGISTER_TILE_N];

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // Prologue：同步加载第一块到 buffer 0。
    float4 firstA =
        loadFloat4OrZero(
            A,
            blockRow + aVectorRow,
            aVectorCol,
            M,
            K
        );

    float4 firstB =
        loadFloat4OrZero(
            B,
            bVectorRow,
            blockCol + bVectorCol,
            K,
            N
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

    for (int tileK = 0; tileK < K; tileK += REGISTER_BLOCK_K) {
        int nextTileK = tileK + REGISTER_BLOCK_K;
        int nextBuffer = currentBuffer ^ 1;
        bool hasNextTile = nextTileK < K;

        float4 nextA =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 nextB =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Prefetch：发出下一块 global load。其结果暂存在寄存器中，
        // 当前块的计算不依赖这些寄存器，可用于隐藏部分访存延迟。
        if (hasNextTile) {
            nextA =
                loadFloat4OrZero(
                    A,
                    blockRow + aVectorRow,
                    nextTileK + aVectorCol,
                    M,
                    K
                );

            nextB =
                loadFloat4OrZero(
                    B,
                    nextTileK + bVectorRow,
                    blockCol + bVectorCol,
                    K,
                    N
                );
        }

        #pragma unroll
        for (int k = 0; k < REGISTER_BLOCK_K; ++k) {
            float regA[REGISTER_TILE_M];
            float regB[REGISTER_TILE_N];

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                regA[i] =
                    tileA[currentBuffer]
                         [i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < REGISTER_TILE_N; ++j) {
                regB[j] =
                    tileB[currentBuffer]
                         [k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REGISTER_TILE_N; ++j) {
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

            // 这次同步同时保证当前块计算结束，以及下一块已经完全
            // 写入 shared memory。下一轮才会覆盖旧的 current buffer。
            __syncthreads();
            currentBuffer = nextBuffer;
        }
    }

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        int row =
            blockRow
            + i * TILE_SIZE
            + threadIdx.y;

        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            int col =
                blockCol
                + j * TILE_SIZE
                + threadIdx.x;

            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


// ============================================================
// V5: cp.async Double Buffered GEMM (Ampere, sm_80+)
//
// cp.async 将下一块直接从 global memory 搬到备用 shared buffer，
// 不经过普通寄存器；当前块的计算与异步复制同时进行。
// ============================================================

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
        && (cols % VECTOR_WIDTH) == 0
        && (col % VECTOR_WIDTH) == 0
        && col + VECTOR_WIDTH <= cols;

    if (canVectorize) {
        size_t offset =
            static_cast<size_t>(row) * cols
            + col;

        unsigned int sharedAddress =
            static_cast<unsigned int>(
                __cvta_generic_to_shared(destination)
            );

        const float* source = matrix + offset;

        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            :
            : "r"(sharedAddress), "l"(source)
        );

        return;
    }
#endif

    // sm_80 以下，或矩阵尾块不满足 16-byte 对齐时同步回退。
    float4 value =
        loadFloat4OrZero(
            matrix,
            row,
            col,
            rows,
            cols
        );

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
    // 每行长度都是 16-byte 的整数倍，使每个 cp.async 目标地址对齐。
    __shared__ __align__(16) float tileA
        [BUFFER_COUNT][REGISTER_BLOCK_M][REGISTER_BLOCK_K];

    __shared__ __align__(16) float tileB
        [BUFFER_COUNT][REGISTER_BLOCK_K][REGISTER_BLOCK_N];

    int blockRow =
        blockIdx.y * REGISTER_BLOCK_M;

    int blockCol =
        blockIdx.x * REGISTER_BLOCK_N;

    int linearThread =
        threadIdx.y * TILE_SIZE
        + threadIdx.x;

    int aVectorRow =
        linearThread
        / (REGISTER_BLOCK_K / VECTOR_WIDTH);

    int aVectorCol =
        (linearThread % (REGISTER_BLOCK_K / VECTOR_WIDTH))
        * VECTOR_WIDTH;

    int bVectorRow =
        linearThread
        / (REGISTER_BLOCK_N / VECTOR_WIDTH);

    int bVectorCol =
        (linearThread % (REGISTER_BLOCK_N / VECTOR_WIDTH))
        * VECTOR_WIDTH;

    float accum[REGISTER_TILE_M][REGISTER_TILE_N];

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // Prologue：异步加载第一个 tile 到 buffer 0，然后等待其就绪。
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

    for (int tileK = 0; tileK < K; tileK += REGISTER_BLOCK_K) {
        int nextTileK = tileK + REGISTER_BLOCK_K;
        int nextBuffer = currentBuffer ^ 1;
        bool hasNextTile = nextTileK < K;

        // 先启动下一块 global -> shared 的异步复制。
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

        // cp.async 在后台填充 next buffer，同时计算 current buffer。
        #pragma unroll
        for (int k = 0; k < REGISTER_BLOCK_K; ++k) {
            float regA[REGISTER_TILE_M];
            float regB[REGISTER_TILE_N];

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                regA[i] =
                    tileA[currentBuffer]
                         [i * TILE_SIZE + threadIdx.y][k];
            }

            #pragma unroll
            for (int j = 0; j < REGISTER_TILE_N; ++j) {
                regB[j] =
                    tileB[currentBuffer]
                         [k][j * TILE_SIZE + threadIdx.x];
            }

            #pragma unroll
            for (int i = 0; i < REGISTER_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REGISTER_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        if (hasNextTile) {
            // 到切换点才等待异步复制，并让整个 block 看见新 tile。
            waitForAsyncCopies();
            __syncthreads();
            currentBuffer = nextBuffer;
        }
    }

    #pragma unroll
    for (int i = 0; i < REGISTER_TILE_M; ++i) {
        int row =
            blockRow
            + i * TILE_SIZE
            + threadIdx.y;

        #pragma unroll
        for (int j = 0; j < REGISTER_TILE_N; ++j) {
            int col =
                blockCol
                + j * TILE_SIZE
                + threadIdx.x;

            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


// ============================================================
// V6: Warp Tiling GEMM
//
// 一个 block 的 4 个 warp 分别计算一个 32 x 32 warp tile。
// 每个 lane 用寄存器累加一个 8 x 4 thread tile。
// ============================================================

constexpr int WARP_SIZE_VALUE = 32;
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;
constexpr int WARPS_PER_BLOCK_M = REGISTER_BLOCK_M / WARP_TILE_M;
constexpr int WARPS_PER_BLOCK_N = REGISTER_BLOCK_N / WARP_TILE_N;
constexpr int WARPS_PER_BLOCK = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N;
constexpr int WARP_BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE_VALUE;
constexpr int WARP_LANES_M = 4;
constexpr int WARP_LANES_N = 8;
constexpr int WARP_THREAD_TILE_M = WARP_TILE_M / WARP_LANES_M;
constexpr int WARP_THREAD_TILE_N = WARP_TILE_N / WARP_LANES_N;

__global__ void gemmWarpTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float tileA
        [REGISTER_BLOCK_M][REGISTER_BLOCK_K + 1];

    __shared__ float tileB
        [REGISTER_BLOCK_K][REGISTER_BLOCK_N];

    int blockRow =
        blockIdx.y * REGISTER_BLOCK_M;

    int blockCol =
        blockIdx.x * REGISTER_BLOCK_N;

    int linearThread =
        threadIdx.y * WARP_SIZE_VALUE
        + threadIdx.x;

    int warpId = linearThread / WARP_SIZE_VALUE;
    int laneId = linearThread % WARP_SIZE_VALUE;

    // 4 个 warp 按 2 x 2 排列，每个 warp 负责 32 x 32 输出。
    int warpRow = warpId / WARPS_PER_BLOCK_N;
    int warpCol = warpId % WARPS_PER_BLOCK_N;

    // 一个 warp 内按 4 x 8 排列 32 个 lane。
    int laneRow = laneId / WARP_LANES_N;
    int laneCol = laneId % WARP_LANES_N;

    float accum[WARP_THREAD_TILE_M][WARP_THREAD_TILE_N];

    #pragma unroll
    for (int i = 0; i < WARP_THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_THREAD_TILE_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    constexpr int aVectorsPerTile =
        REGISTER_BLOCK_M * REGISTER_BLOCK_K / VECTOR_WIDTH;

    constexpr int bVectorsPerTile =
        REGISTER_BLOCK_K * REGISTER_BLOCK_N / VECTOR_WIDTH;

    constexpr int aVectorsPerRow =
        REGISTER_BLOCK_K / VECTOR_WIDTH;

    constexpr int bVectorsPerRow =
        REGISTER_BLOCK_N / VECTOR_WIDTH;

    for (int tileK = 0; tileK < K; tileK += REGISTER_BLOCK_K) {
        // 128 个线程协作加载 64 x 16 的 A tile；
        // 每线程加载两个 float4。
        #pragma unroll
        for (int vectorIndex = linearThread;
             vectorIndex < aVectorsPerTile;
             vectorIndex += WARP_BLOCK_THREADS) {
            int tileRow = vectorIndex / aVectorsPerRow;
            int tileCol =
                (vectorIndex % aVectorsPerRow) * VECTOR_WIDTH;

            float4 value =
                loadFloat4OrZero(
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

        // 同样协作加载 16 x 64 的 B tile。
        #pragma unroll
        for (int vectorIndex = linearThread;
             vectorIndex < bVectorsPerTile;
             vectorIndex += WARP_BLOCK_THREADS) {
            int tileRow = vectorIndex / bVectorsPerRow;
            int tileCol =
                (vectorIndex % bVectorsPerRow) * VECTOR_WIDTH;

            float4 value =
                loadFloat4OrZero(
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
        for (int k = 0; k < REGISTER_BLOCK_K; ++k) {
            float regA[WARP_THREAD_TILE_M];
            float regB[WARP_THREAD_TILE_N];

            // 当前 lane 在 warp tile 中负责 8 行。
            #pragma unroll
            for (int i = 0; i < WARP_THREAD_TILE_M; ++i) {
                int rowInTile =
                    warpRow * WARP_TILE_M
                    + i * WARP_LANES_M
                    + laneRow;

                regA[i] = tileA[rowInTile][k];
            }

            // 当前 lane 在 warp tile 中负责 4 列。
            #pragma unroll
            for (int j = 0; j < WARP_THREAD_TILE_N; ++j) {
                int colInTile =
                    warpCol * WARP_TILE_N
                    + j * WARP_LANES_N
                    + laneCol;

                regB[j] = tileB[k][colInTile];
            }

            // 8 x 4 外积：每个 k 执行 32 次 FMA。
            #pragma unroll
            for (int i = 0; i < WARP_THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < WARP_THREAD_TILE_N; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < WARP_THREAD_TILE_M; ++i) {
        int row =
            blockRow
            + warpRow * WARP_TILE_M
            + i * WARP_LANES_M
            + laneRow;

        #pragma unroll
        for (int j = 0; j < WARP_THREAD_TILE_N; ++j) {
            int col =
                blockCol
                + warpCol * WARP_TILE_N
                + j * WARP_LANES_N
                + laneCol;

            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}


// ============================================================
// V7: Tensor Core GEMM (WMMA)
//
// 输入使用 FP16，累加与输出使用 FP32。一个 block 仍然计算
// 64 x 64 的 C tile；4 个 warp 各负责一个 32 x 32 warp tile。
// 每个 32 x 32 warp tile 由 4 个 16 x 16 WMMA fragment 组成。
// ============================================================

constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_N = 16;
constexpr int WMMA_TILE_K = 16;
constexpr int TENSOR_WARP_TILE_M = 32;
constexpr int TENSOR_WARP_TILE_N = 32;
constexpr int TENSOR_BLOCK_M = 64;
constexpr int TENSOR_BLOCK_N = 64;
constexpr int TENSOR_BLOCK_K = WMMA_TILE_K;
constexpr int TENSOR_WARPS_M = TENSOR_BLOCK_M / TENSOR_WARP_TILE_M;
constexpr int TENSOR_WARPS_N = TENSOR_BLOCK_N / TENSOR_WARP_TILE_N;
constexpr int TENSOR_WARPS_PER_BLOCK = TENSOR_WARPS_M * TENSOR_WARPS_N;
constexpr int TENSOR_BLOCK_THREADS = TENSOR_WARPS_PER_BLOCK * WARP_SIZE_VALUE;
constexpr int TENSOR_FRAGMENTS_M = TENSOR_WARP_TILE_M / WMMA_TILE_M;
constexpr int TENSOR_FRAGMENTS_N = TENSOR_WARP_TILE_N / WMMA_TILE_N;
constexpr float TENSOR_CORE_ERROR_TOLERANCE = 1e-2f;

static_assert(TENSOR_WARPS_PER_BLOCK == 4);
static_assert(TENSOR_BLOCK_K == 16);

__global__ void gemmTensorCore(
    const __half* A,
    const __half* B,
    float* C,
    int M,
    int N,
    int K)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;

    // WMMA load_matrix_sync 要求地址和 leading dimension 满足对齐要求。
    // 这里的行跨度均为 16 或 64 个 half，且数组基址按 32 byte 对齐。
    __shared__ __align__(32) __half tileA[TENSOR_BLOCK_M][TENSOR_BLOCK_K];
    __shared__ __align__(32) __half tileB[TENSOR_BLOCK_K][TENSOR_BLOCK_N];

    // 只在矩阵边界无法直接写回时使用。
    __shared__ __align__(32) float outputTile[TENSOR_BLOCK_M][TENSOR_BLOCK_N];

    int linearThread =
        threadIdx.y * blockDim.x
        + threadIdx.x;

    int warpId = linearThread / WARP_SIZE_VALUE;
    int warpRow = warpId / TENSOR_WARPS_N;
    int warpCol = warpId % TENSOR_WARPS_N;

    int blockRow = blockIdx.y * TENSOR_BLOCK_M;
    int blockCol = blockIdx.x * TENSOR_BLOCK_N;

    wmma::fragment<
        wmma::accumulator,
        WMMA_TILE_M,
        WMMA_TILE_N,
        WMMA_TILE_K,
        float
    > accum[TENSOR_FRAGMENTS_M][TENSOR_FRAGMENTS_N];

    #pragma unroll
    for (int i = 0; i < TENSOR_FRAGMENTS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < TENSOR_FRAGMENTS_N; ++j) {
            wmma::fill_fragment(accum[i][j], 0.0f);
        }
    }

    for (int tileK = 0; tileK < K; tileK += TENSOR_BLOCK_K) {
        // 128 个线程协作加载 64 x 16 的 FP16 A tile。
        for (int index = linearThread;
             index < TENSOR_BLOCK_M * TENSOR_BLOCK_K;
             index += TENSOR_BLOCK_THREADS) {
            int tileRow = index / TENSOR_BLOCK_K;
            int tileCol = index % TENSOR_BLOCK_K;
            int globalRow = blockRow + tileRow;
            int globalCol = tileK + tileCol;

            tileA[tileRow][tileCol] =
                (globalRow < M && globalCol < K)
                    ? A[static_cast<size_t>(globalRow) * K + globalCol]
                    : __float2half(0.0f);
        }

        // 同样协作加载 16 x 64 的 FP16 B tile。
        for (int index = linearThread;
             index < TENSOR_BLOCK_K * TENSOR_BLOCK_N;
             index += TENSOR_BLOCK_THREADS) {
            int tileRow = index / TENSOR_BLOCK_N;
            int tileCol = index % TENSOR_BLOCK_N;
            int globalRow = tileK + tileRow;
            int globalCol = blockCol + tileCol;

            tileB[tileRow][tileCol] =
                (globalRow < K && globalCol < N)
                    ? B[static_cast<size_t>(globalRow) * N + globalCol]
                    : __float2half(0.0f);
        }

        __syncthreads();

        wmma::fragment<
            wmma::matrix_a,
            WMMA_TILE_M,
            WMMA_TILE_N,
            WMMA_TILE_K,
            __half,
            wmma::row_major
        > fragmentA[TENSOR_FRAGMENTS_M];

        wmma::fragment<
            wmma::matrix_b,
            WMMA_TILE_M,
            WMMA_TILE_N,
            WMMA_TILE_K,
            __half,
            wmma::row_major
        > fragmentB[TENSOR_FRAGMENTS_N];

        #pragma unroll
        for (int i = 0; i < TENSOR_FRAGMENTS_M; ++i) {
            int row =
                warpRow * TENSOR_WARP_TILE_M
                + i * WMMA_TILE_M;

            wmma::load_matrix_sync(
                fragmentA[i],
                &tileA[row][0],
                TENSOR_BLOCK_K
            );
        }

        #pragma unroll
        for (int j = 0; j < TENSOR_FRAGMENTS_N; ++j) {
            int col =
                warpCol * TENSOR_WARP_TILE_N
                + j * WMMA_TILE_N;

            wmma::load_matrix_sync(
                fragmentB[j],
                &tileB[0][col],
                TENSOR_BLOCK_N
            );
        }

        // 每次 mma_sync 由整个 warp 协作完成 16 x 16 x 16 的矩阵乘加。
        #pragma unroll
        for (int i = 0; i < TENSOR_FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < TENSOR_FRAGMENTS_N; ++j) {
                wmma::mma_sync(
                    accum[i][j],
                    fragmentA[i],
                    fragmentB[j],
                    accum[i][j]
                );
            }
        }

        // 所有 warp 用完当前 shared-memory tile 后才能覆盖它。
        __syncthreads();
    }

    int warpOutputRow = warpRow * TENSOR_WARP_TILE_M;
    int warpOutputCol = warpCol * TENSOR_WARP_TILE_N;

    bool directStore =
        blockRow + TENSOR_BLOCK_M <= M
        && blockCol + TENSOR_BLOCK_N <= N
        && N % 4 == 0;

    if (directStore) {
        // 完整 tile 直接由 WMMA 从寄存器 fragment 写到 global memory。
        #pragma unroll
        for (int i = 0; i < TENSOR_FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < TENSOR_FRAGMENTS_N; ++j) {
                float* destination =
                    C
                    + static_cast<size_t>(
                        blockRow + warpOutputRow + i * WMMA_TILE_M
                    ) * N
                    + blockCol + warpOutputCol + j * WMMA_TILE_N;

                wmma::store_matrix_sync(
                    destination,
                    accum[i][j],
                    N,
                    wmma::mem_row_major
                );
            }
        }
    } else {
        // 边界 tile 先完整写入 shared memory，再由线程做带边界判断的写回。
        #pragma unroll
        for (int i = 0; i < TENSOR_FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < TENSOR_FRAGMENTS_N; ++j) {
                wmma::store_matrix_sync(
                    &outputTile
                        [warpOutputRow + i * WMMA_TILE_M]
                        [warpOutputCol + j * WMMA_TILE_N],
                    accum[i][j],
                    TENSOR_BLOCK_N,
                    wmma::mem_row_major
                );
            }
        }

        __syncthreads();

        for (int index = linearThread;
             index < TENSOR_BLOCK_M * TENSOR_BLOCK_N;
             index += TENSOR_BLOCK_THREADS) {
            int tileRow = index / TENSOR_BLOCK_N;
            int tileCol = index % TENSOR_BLOCK_N;
            int row = blockRow + tileRow;
            int col = blockCol + tileCol;

            if (row < M && col < N) {
                C[static_cast<size_t>(row) * N + col] =
                    outputTile[tileRow][tileCol];
            }
        }
    }
#else
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
#endif
}

float benchmarkNaive(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmNaive<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmNaive<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return totalMs / repeat;
}

float benchmarkSharedMemory(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmSharedMemory<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmSharedMemory<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkRegisterTiled(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
        (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmRegisterTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmRegisterTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkVectorizedRegisterTiled(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
        (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmVectorizedRegisterTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmVectorizedRegisterTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkDoubleBuffered(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
        (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmDoubleBuffered<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmDoubleBuffered<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkCpAsyncDoubleBuffered(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(TILE_SIZE, TILE_SIZE);

    dim3 grid(
        (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
        (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmCpAsyncDoubleBuffered<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmCpAsyncDoubleBuffered<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkWarpTiled(
    const float* dA,
    const float* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(WARP_SIZE_VALUE, WARPS_PER_BLOCK);

    dim3 grid(
        (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
        (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmWarpTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmWarpTiled<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

float benchmarkTensorCore(
    const __half* dA,
    const __half* dB,
    float* dC,
    int M,
    int N,
    int K,
    int repeat = 20)
{
    dim3 block(WARP_SIZE_VALUE, TENSOR_WARPS_PER_BLOCK);

    dim3 grid(
        (N + TENSOR_BLOCK_N - 1) / TENSOR_BLOCK_N,
        (M + TENSOR_BLOCK_M - 1) / TENSOR_BLOCK_M
    );

    // Warmup（预热）
    for (int i = 0; i < 5; ++i) {
        gemmTensorCore<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        gemmTensorCore<<<grid, block>>>(
            dA, dB, dC,
            M, N, K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &totalMs,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return totalMs / repeat;
}

int main()
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    int device = 0;
    cudaDeviceProp deviceProperties{};
    cudaFuncAttributes cpAsyncAttributes{};
    cudaFuncAttributes tensorCoreAttributes{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
    CUDA_CHECK(
        cudaFuncGetAttributes(
            &cpAsyncAttributes,
            gemmCpAsyncDoubleBuffered
        )
    );
    CUDA_CHECK(
        cudaFuncGetAttributes(
            &tensorCoreAttributes,
            gemmTensorCore
        )
    );

    bool cpAsyncAvailable =
        deviceProperties.major >= 8
        && cpAsyncAttributes.binaryVersion >= 80;

    bool tensorCoreAvailable =
        deviceProperties.major >= 7
        && tensorCoreAttributes.binaryVersion >= 70;

    size_t sizeA =
        static_cast<size_t>(M) * K;

    size_t sizeB =
        static_cast<size_t>(K) * N;

    size_t sizeC =
        static_cast<size_t>(M) * N;


    std::vector<float> hA(sizeA);
    std::vector<float> hB(sizeB);
    std::vector<__half> hAHalf(sizeA);
    std::vector<__half> hBHalf(sizeB);
    std::vector<float> hCNaive(sizeC);
    std::vector<float> hCShared(sizeC);
    std::vector<float> hCRegister(sizeC);
    std::vector<float> hCVectorized(sizeC);
    std::vector<float> hCDoubleBuffered(sizeC);
    std::vector<float> hCCpAsync(sizeC);
    std::vector<float> hCWarpTiled(sizeC);
    std::vector<float> hCTensorCore(sizeC);


    // 初始化
    for (size_t i = 0; i < sizeA; ++i) {
        hA[i] =
            static_cast<float>(
                static_cast<int>(i % 7) - 3
            ) * 0.1f;

        hAHalf[i] = __float2half(hA[i]);
    }

    for (size_t i = 0; i < sizeB; ++i) {
        hB[i] =
            static_cast<float>(
                static_cast<int>(i % 5) - 2
            ) * 0.1f;

        hBHalf[i] = __float2half(hB[i]);
    }


    float* dA;
    float* dB;
    __half* dAHalf;
    __half* dBHalf;
    float* dCNaive;
    float* dCShared;
    float* dCRegister;
    float* dCVectorized;
    float* dCDoubleBuffered;
    float* dCCpAsync;
    float* dCWarpTiled;
    float* dCTensorCore;

    CUDA_CHECK(
        cudaMalloc(&dA, sizeA * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dB, sizeB * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dAHalf, sizeA * sizeof(__half))
    );

    CUDA_CHECK(
        cudaMalloc(&dBHalf, sizeB * sizeof(__half))
    );

    CUDA_CHECK(
        cudaMalloc(&dCNaive, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCShared, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCRegister, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCVectorized, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCDoubleBuffered, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCCpAsync, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCWarpTiled, sizeC * sizeof(float))
    );

    CUDA_CHECK(
        cudaMalloc(&dCTensorCore, sizeC * sizeof(float))
    );


    CUDA_CHECK(
        cudaMemcpy(
            dA,
            hA.data(),
            sizeA * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dB,
            hB.data(),
            sizeB * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dAHalf,
            hAHalf.data(),
            sizeA * sizeof(__half),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            dBHalf,
            hBHalf.data(),
            sizeB * sizeof(__half),
            cudaMemcpyHostToDevice
        )
    );


    float naiveMs =
        benchmarkNaive(
            dA,
            dB,
            dCNaive,
            M,
            N,
            K
        );

    float sharedMs =
        benchmarkSharedMemory(
            dA,
            dB,
            dCShared,
            M,
            N,
            K
        );

    float registerMs =
        benchmarkRegisterTiled(
            dA,
            dB,
            dCRegister,
            M,
            N,
            K
        );

    float vectorizedMs =
        benchmarkVectorizedRegisterTiled(
            dA,
            dB,
            dCVectorized,
            M,
            N,
            K
        );

    float doubleBufferedMs =
        benchmarkDoubleBuffered(
            dA,
            dB,
            dCDoubleBuffered,
            M,
            N,
            K
        );

    float cpAsyncMs = 0.0f;

    if (cpAsyncAvailable) {
        cpAsyncMs =
            benchmarkCpAsyncDoubleBuffered(
                dA,
                dB,
                dCCpAsync,
                M,
                N,
                K
            );
    }

    float warpTiledMs =
        benchmarkWarpTiled(
            dA,
            dB,
            dCWarpTiled,
            M,
            N,
            K
        );

    float tensorCoreMs = 0.0f;

    if (tensorCoreAvailable) {
        tensorCoreMs =
            benchmarkTensorCore(
                dAHalf,
                dBHalf,
                dCTensorCore,
                M,
                N,
                K
            );
    }


    double flops =
        2.0
        * M
        * N
        * K;


    double naiveGflops =
        flops
        /
        (naiveMs / 1000.0)
        /
        1e9;

    double sharedGflops =
        flops
        /
        (sharedMs / 1000.0)
        /
        1e9;

    double registerGflops =
        flops
        /
        (registerMs / 1000.0)
        /
        1e9;

    double vectorizedGflops =
        flops
        /
        (vectorizedMs / 1000.0)
        /
        1e9;

    double doubleBufferedGflops =
        flops
        /
        (doubleBufferedMs / 1000.0)
        /
        1e9;

    double cpAsyncGflops =
        cpAsyncAvailable
            ? flops / (cpAsyncMs / 1000.0) / 1e9
            : 0.0;

    double warpTiledGflops =
        flops
        /
        (warpTiledMs / 1000.0)
        /
        1e9;

    double tensorCoreGflops =
        tensorCoreAvailable
            ? flops / (tensorCoreMs / 1000.0) / 1e9
            : 0.0;


    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "=================================\n";

    std::cout
        << "V0 Naive GEMM\n";

    std::cout
        << "Matrix: "
        << M << " x "
        << N << " x "
        << K << "\n";

    std::cout
        << "Time: "
        << naiveMs
        << " ms\n";

    std::cout
        << "Performance: "
        << naiveGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V1 Shared Memory Tiling GEMM\n";

    std::cout
        << "Tile: "
        << TILE_SIZE << " x "
        << TILE_SIZE << "\n";

    std::cout
        << "Time: "
        << sharedMs
        << " ms\n";

    std::cout
        << "Performance: "
        << sharedGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V2 Register Tiling GEMM\n";

    std::cout
        << "Block tile: "
        << REGISTER_BLOCK_M << " x "
        << REGISTER_BLOCK_N << " x "
        << REGISTER_BLOCK_K << "\n";

    std::cout
        << "Thread tile: "
        << REGISTER_TILE_M << " x "
        << REGISTER_TILE_N << "\n";

    std::cout
        << "Time: "
        << registerMs
        << " ms\n";

    std::cout
        << "Performance: "
        << registerGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V3 Vectorized Load GEMM\n";

    std::cout
        << "Block tile: "
        << REGISTER_BLOCK_M << " x "
        << REGISTER_BLOCK_N << " x "
        << REGISTER_BLOCK_K << "\n";

    std::cout
        << "Thread tile: "
        << REGISTER_TILE_M << " x "
        << REGISTER_TILE_N << "\n";

    std::cout
        << "Vector width: float"
        << VECTOR_WIDTH << "\n";

    std::cout
        << "Time: "
        << vectorizedMs
        << " ms\n";

    std::cout
        << "Performance: "
        << vectorizedGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V4 Double Buffered GEMM\n";

    std::cout
        << "Block tile: "
        << REGISTER_BLOCK_M << " x "
        << REGISTER_BLOCK_N << " x "
        << REGISTER_BLOCK_K << "\n";

    std::cout
        << "Buffers: "
        << BUFFER_COUNT << "\n";

    std::cout
        << "Time: "
        << doubleBufferedMs
        << " ms\n";

    std::cout
        << "Performance: "
        << doubleBufferedGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V5 cp.async Double Buffered GEMM\n";

    if (cpAsyncAvailable) {
        std::cout
            << "Block tile: "
            << REGISTER_BLOCK_M << " x "
            << REGISTER_BLOCK_N << " x "
            << REGISTER_BLOCK_K << "\n";

        std::cout
            << "Time: "
            << cpAsyncMs
            << " ms\n";

        std::cout
            << "Performance: "
            << cpAsyncGflops
            << " GFLOPS\n";
    } else {
        std::cout
            << "Status: SKIPPED "
            << "(requires sm_80+ device and build)\n";
    }

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V6 Warp Tiling GEMM\n";

    std::cout
        << "Block tile: "
        << REGISTER_BLOCK_M << " x "
        << REGISTER_BLOCK_N << " x "
        << REGISTER_BLOCK_K << "\n";

    std::cout
        << "Warp tile: "
        << WARP_TILE_M << " x "
        << WARP_TILE_N << "\n";

    std::cout
        << "Thread tile: "
        << WARP_THREAD_TILE_M << " x "
        << WARP_THREAD_TILE_N << "\n";

    std::cout
        << "Warps per block: "
        << WARPS_PER_BLOCK << "\n";

    std::cout
        << "Time: "
        << warpTiledMs
        << " ms\n";

    std::cout
        << "Performance: "
        << warpTiledGflops
        << " GFLOPS\n";

    std::cout
        << "---------------------------------\n";

    std::cout
        << "V7 Tensor Core GEMM (WMMA)\n";

    if (tensorCoreAvailable) {
        std::cout
            << "Input / accumulator: FP16 / FP32\n";

        std::cout
            << "Block tile: "
            << TENSOR_BLOCK_M << " x "
            << TENSOR_BLOCK_N << " x "
            << TENSOR_BLOCK_K << "\n";

        std::cout
            << "Warp tile: "
            << TENSOR_WARP_TILE_M << " x "
            << TENSOR_WARP_TILE_N << "\n";

        std::cout
            << "WMMA tile: "
            << WMMA_TILE_M << " x "
            << WMMA_TILE_N << " x "
            << WMMA_TILE_K << "\n";

        std::cout
            << "Time: "
            << tensorCoreMs
            << " ms\n";

        std::cout
            << "Performance: "
            << tensorCoreGflops
            << " GFLOPS\n";
    } else {
        std::cout
            << "Status: SKIPPED "
            << "(requires sm_70+ device and build)\n";
    }


    CUDA_CHECK(
        cudaMemcpy(
            hCNaive.data(),
            dCNaive,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            hCShared.data(),
            dCShared,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            hCRegister.data(),
            dCRegister,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            hCVectorized.data(),
            dCVectorized,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            hCDoubleBuffered.data(),
            dCDoubleBuffered,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    if (cpAsyncAvailable) {
        CUDA_CHECK(
            cudaMemcpy(
                hCCpAsync.data(),
                dCCpAsync,
                sizeC * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );
    }

    CUDA_CHECK(
        cudaMemcpy(
            hCWarpTiled.data(),
            dCWarpTiled,
            sizeC * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    if (tensorCoreAvailable) {
        CUDA_CHECK(
            cudaMemcpy(
                hCTensorCore.data(),
                dCTensorCore,
                sizeC * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );
    }

    float maxAbsErrorShared = 0.0f;
    float maxAbsErrorRegister = 0.0f;
    float maxAbsErrorVectorized = 0.0f;
    float maxAbsErrorDoubleBuffered = 0.0f;
    float maxAbsErrorCpAsync = 0.0f;
    float maxAbsErrorWarpTiled = 0.0f;
    float maxAbsErrorTensorCore = 0.0f;

    for (size_t i = 0; i < sizeC; ++i) {
        maxAbsErrorShared =
            std::fmax(
                maxAbsErrorShared,
                std::fabs(hCNaive[i] - hCShared[i])
            );

        maxAbsErrorRegister =
            std::fmax(
                maxAbsErrorRegister,
                std::fabs(hCNaive[i] - hCRegister[i])
            );

        maxAbsErrorVectorized =
            std::fmax(
                maxAbsErrorVectorized,
                std::fabs(hCNaive[i] - hCVectorized[i])
            );

        maxAbsErrorDoubleBuffered =
            std::fmax(
                maxAbsErrorDoubleBuffered,
                std::fabs(hCNaive[i] - hCDoubleBuffered[i])
            );

        if (cpAsyncAvailable) {
            maxAbsErrorCpAsync =
                std::fmax(
                    maxAbsErrorCpAsync,
                    std::fabs(hCNaive[i] - hCCpAsync[i])
                );
        }

        maxAbsErrorWarpTiled =
            std::fmax(
                maxAbsErrorWarpTiled,
                std::fabs(hCNaive[i] - hCWarpTiled[i])
            );

        if (tensorCoreAvailable) {
            maxAbsErrorTensorCore =
                std::fmax(
                    maxAbsErrorTensorCore,
                    std::fabs(hCNaive[i] - hCTensorCore[i])
                );
        }
    }

    bool passed =
        maxAbsErrorShared < 1e-3f
        && maxAbsErrorRegister < 1e-3f
        && maxAbsErrorVectorized < 1e-3f
        && maxAbsErrorDoubleBuffered < 1e-3f
        && (!cpAsyncAvailable || maxAbsErrorCpAsync < 1e-3f)
        && maxAbsErrorWarpTiled < 1e-3f
        && (!tensorCoreAvailable
            || maxAbsErrorTensorCore < TENSOR_CORE_ERROR_TOLERANCE);

    std::cout
        << "V1 max absolute error: "
        << maxAbsErrorShared << "\n";

    std::cout
        << "V2 max absolute error: "
        << maxAbsErrorRegister << "\n";

    std::cout
        << "V3 max absolute error: "
        << maxAbsErrorVectorized << "\n";

    std::cout
        << "V4 max absolute error: "
        << maxAbsErrorDoubleBuffered << "\n";

    if (cpAsyncAvailable) {
        std::cout
            << "V5 max absolute error: "
            << maxAbsErrorCpAsync << "\n";
    }

    std::cout
        << "V6 max absolute error: "
        << maxAbsErrorWarpTiled << "\n";

    if (tensorCoreAvailable) {
        std::cout
            << "V7 max absolute error: "
            << maxAbsErrorTensorCore
            << " (FP16 tolerance: "
            << TENSOR_CORE_ERROR_TOLERANCE
            << ")\n";
    }

    std::cout
        << "Check: "
        << (passed ? "PASSED" : "FAILED")
        << "\n";

    std::cout
        << "=================================\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dAHalf));
    CUDA_CHECK(cudaFree(dBHalf));
    CUDA_CHECK(cudaFree(dCNaive));
    CUDA_CHECK(cudaFree(dCShared));
    CUDA_CHECK(cudaFree(dCRegister));
    CUDA_CHECK(cudaFree(dCVectorized));
    CUDA_CHECK(cudaFree(dCDoubleBuffered));
    CUDA_CHECK(cudaFree(dCCpAsync));
    CUDA_CHECK(cudaFree(dCWarpTiled));
    CUDA_CHECK(cudaFree(dCTensorCore));

    return passed ? 0 : EXIT_FAILURE;
}
