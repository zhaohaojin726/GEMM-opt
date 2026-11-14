#include <cuda_fp16.h>
#include <mma.h>

#include "../include/gemm_benchmark.cuh"

constexpr int WARP_SIZE_VALUE = 32;
constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_N = 16;
constexpr int WMMA_TILE_K = 16;
constexpr int WARP_TILE_M = 32;
constexpr int WARP_TILE_N = 32;
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = WMMA_TILE_K;
constexpr int WARPS_M = BLOCK_TILE_M / WARP_TILE_M;
constexpr int WARPS_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr int BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE_VALUE;
constexpr int FRAGMENTS_M = WARP_TILE_M / WMMA_TILE_M;
constexpr int FRAGMENTS_N = WARP_TILE_N / WMMA_TILE_N;

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

    __shared__ __align__(32) __half tileA[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ __align__(32) __half tileB[BLOCK_TILE_K][BLOCK_TILE_N];
    __shared__ __align__(32) float outputTile[BLOCK_TILE_M][BLOCK_TILE_N];

    int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = linearThread / WARP_SIZE_VALUE;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int blockRow = blockIdx.y * BLOCK_TILE_M;
    int blockCol = blockIdx.x * BLOCK_TILE_N;

    wmma::fragment<
        wmma::accumulator,
        WMMA_TILE_M,
        WMMA_TILE_N,
        WMMA_TILE_K,
        float
    > accum[FRAGMENTS_M][FRAGMENTS_N];

    #pragma unroll
    for (int i = 0; i < FRAGMENTS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGMENTS_N; ++j) {
            wmma::fill_fragment(accum[i][j], 0.0f);
        }
    }

    for (int tileK = 0; tileK < K; tileK += BLOCK_TILE_K) {
        for (int index = linearThread;
             index < BLOCK_TILE_M * BLOCK_TILE_K;
             index += BLOCK_THREADS) {
            int tileRow = index / BLOCK_TILE_K;
            int tileCol = index % BLOCK_TILE_K;
            int row = blockRow + tileRow;
            int col = tileK + tileCol;

            tileA[tileRow][tileCol] =
                (row < M && col < K)
                    ? A[static_cast<size_t>(row) * K + col]
                    : __float2half(0.0f);
        }

        for (int index = linearThread;
             index < BLOCK_TILE_K * BLOCK_TILE_N;
             index += BLOCK_THREADS) {
            int tileRow = index / BLOCK_TILE_N;
            int tileCol = index % BLOCK_TILE_N;
            int row = tileK + tileRow;
            int col = blockCol + tileCol;

            tileB[tileRow][tileCol] =
                (row < K && col < N)
                    ? B[static_cast<size_t>(row) * N + col]
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
        > fragmentA[FRAGMENTS_M];

        wmma::fragment<
            wmma::matrix_b,
            WMMA_TILE_M,
            WMMA_TILE_N,
            WMMA_TILE_K,
            __half,
            wmma::row_major
        > fragmentB[FRAGMENTS_N];

        #pragma unroll
        for (int i = 0; i < FRAGMENTS_M; ++i) {
            int row = warpRow * WARP_TILE_M + i * WMMA_TILE_M;
            wmma::load_matrix_sync(fragmentA[i], &tileA[row][0], BLOCK_TILE_K);
        }

        #pragma unroll
        for (int j = 0; j < FRAGMENTS_N; ++j) {
            int col = warpCol * WARP_TILE_N + j * WMMA_TILE_N;
            wmma::load_matrix_sync(fragmentB[j], &tileB[0][col], BLOCK_TILE_N);
        }

        #pragma unroll
        for (int i = 0; i < FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGMENTS_N; ++j) {
                wmma::mma_sync(
                    accum[i][j],
                    fragmentA[i],
                    fragmentB[j],
                    accum[i][j]
                );
            }
        }

        __syncthreads();
    }

    int warpOutputRow = warpRow * WARP_TILE_M;
    int warpOutputCol = warpCol * WARP_TILE_N;

    bool directStore =
        blockRow + BLOCK_TILE_M <= M
        && blockCol + BLOCK_TILE_N <= N
        && N % 4 == 0;

    if (directStore) {
        #pragma unroll
        for (int i = 0; i < FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGMENTS_N; ++j) {
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
        #pragma unroll
        for (int i = 0; i < FRAGMENTS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGMENTS_N; ++j) {
                wmma::store_matrix_sync(
                    &outputTile
                        [warpOutputRow + i * WMMA_TILE_M]
                        [warpOutputCol + j * WMMA_TILE_N],
                    accum[i][j],
                    BLOCK_TILE_N,
                    wmma::mem_row_major
                );
            }
        }

        __syncthreads();

        for (int index = linearThread;
             index < BLOCK_TILE_M * BLOCK_TILE_N;
             index += BLOCK_THREADS) {
            int tileRow = index / BLOCK_TILE_N;
            int tileCol = index % BLOCK_TILE_N;
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

int main()
{
    int device = 0;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, gemmTensorCore));

    if (properties.major < 7 || attributes.binaryVersion < 70) {
        std::cout
            << "V7 Tensor Core GEMM (WMMA)\n"
            << "Status: SKIPPED (requires sm_70+ device and build)\n";
        return EXIT_SUCCESS;
    }

    auto launch = [](
        const __half* A,
        const __half* B,
        float* C,
        int M,
        int N,
        int K) {
        dim3 block(WARP_SIZE_VALUE, WARPS_PER_BLOCK);
        dim3 grid(
            (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
            (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M
        );

        gemmTensorCore<<<grid, block>>>(A, B, C, M, N, K);
    };

    return gemm_demo::runVersion<__half>(
        "V7 Tensor Core GEMM (WMMA)",
        "Input / accumulator: FP16 / FP32\nBlock tile: 64 x 64 x 16\n"
        "Warp tile: 32 x 32\nWMMA tile: 16 x 16 x 16\n",
        [] (float value) { return __float2half(value); },
        launch,
        1e-2f
    );
}
