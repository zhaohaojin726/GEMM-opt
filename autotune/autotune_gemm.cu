#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t error__ = (call);                                             \
    if (error__ != cudaSuccess) {                                             \
        std::cerr << "CUDA error: " << cudaGetErrorString(error__)           \
                  << " at " << __FILE__ << ':' << __LINE__ << '\n';          \
        std::exit(EXIT_FAILURE);                                              \
    }                                                                         \
} while (0)

#define CUBLAS_CHECK(call)                                                   \
do {                                                                         \
    cublasStatus_t status__ = (call);                                         \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                  \
        std::cerr << "cuBLAS error: status=" << static_cast<int>(status__)   \
                  << " at " << __FILE__ << ':' << __LINE__ << '\n';          \
        std::exit(EXIT_FAILURE);                                              \
    }                                                                         \
} while (0)

namespace {

constexpr int kVectorWidth = 4;
constexpr int kDefaultWarmups = 5;

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

    if (col + kVectorWidth <= cols) {
        const size_t offset = static_cast<size_t>(row) * cols + col;
        if ((offset & (kVectorWidth - 1)) == 0) {
            return *reinterpret_cast<const float4*>(matrix + offset);
        }
    }

    if (col < cols) value.x = matrix[static_cast<size_t>(row) * cols + col];
    if (col + 1 < cols) value.y = matrix[static_cast<size_t>(row) * cols + col + 1];
    if (col + 2 < cols) value.z = matrix[static_cast<size_t>(row) * cols + col + 2];
    if (col + 3 < cols) value.w = matrix[static_cast<size_t>(row) * cols + col + 3];
    return value;
}

// BM/BN 是一个 thread block 计算的 C 子块，BK 是一次搬入 shared memory
// 后沿 K 方向累计的长度，TM/TN 是单线程保存在寄存器中的 C 子块。
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemmTunable(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{
    static_assert(BM % TM == 0 && BN % TN == 0);
    static_assert(BK % kVectorWidth == 0);
    constexpr int kBlockThreads = (BM / TM) * (BN / TN);
    static_assert(kBlockThreads > 0 && kBlockThreads <= 1024);

    __shared__ float tileA[BM][BK + 1];
    __shared__ float tileB[BK][BN];

    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;
    const int linearThread = threadRow * blockDim.x + threadCol;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float accum[TM][TN] = {};

    constexpr int kAVectors = BM * BK / kVectorWidth;
    constexpr int kBVectors = BK * BN / kVectorWidth;
    constexpr int kAVectorsPerRow = BK / kVectorWidth;
    constexpr int kBVectorsPerRow = BN / kVectorWidth;

    for (int tileK = 0; tileK < K; tileK += BK) {
        for (int vectorIndex = linearThread;
             vectorIndex < kAVectors;
             vectorIndex += kBlockThreads) {
            const int tileRow = vectorIndex / kAVectorsPerRow;
            const int tileCol =
                (vectorIndex % kAVectorsPerRow) * kVectorWidth;
            const float4 value = loadFloat4OrZero(
                A, blockRow + tileRow, tileK + tileCol, M, K);

            tileA[tileRow][tileCol] = value.x;
            tileA[tileRow][tileCol + 1] = value.y;
            tileA[tileRow][tileCol + 2] = value.z;
            tileA[tileRow][tileCol + 3] = value.w;
        }

        for (int vectorIndex = linearThread;
             vectorIndex < kBVectors;
             vectorIndex += kBlockThreads) {
            const int tileRow = vectorIndex / kBVectorsPerRow;
            const int tileCol =
                (vectorIndex % kBVectorsPerRow) * kVectorWidth;
            const float4 value = loadFloat4OrZero(
                B, tileK + tileRow, blockCol + tileCol, K, N);

            tileB[tileRow][tileCol] = value.x;
            tileB[tileRow][tileCol + 1] = value.y;
            tileB[tileRow][tileCol + 2] = value.z;
            tileB[tileRow][tileCol + 3] = value.w;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = tileA[i * blockDim.y + threadRow][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = tileB[k][j * blockDim.x + threadCol];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] = fmaf(regA[i], regB[j], accum[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + i * blockDim.y + threadRow;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = blockCol + j * blockDim.x + threadCol;
            if (row < M && col < N) {
                C[static_cast<size_t>(row) * N + col] = accum[i][j];
            }
        }
    }
}

using LaunchFn = void (*)(
    const float*, const float*, float*, int, int, int, cudaStream_t);
using MetadataFn = void (*)(
    const cudaDeviceProp&, int&, size_t&, double&, int&);

template <int BM, int BN, int BK, int TM, int TN>
void launchKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    constexpr int kBlockX = BN / TN;
    constexpr int kBlockY = BM / TM;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemmTunable<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

template <int BM, int BN, int BK, int TM, int TN>
void queryMetadata(
    const cudaDeviceProp& properties,
    int& registersPerThread,
    size_t& staticSharedBytes,
    double& theoreticalOccupancyPct,
    int& blockThreads)
{
    constexpr int kBlockThreads = (BM / TM) * (BN / TN);
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, gemmTunable<BM, BN, BK, TM, TN>));

    int activeBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerSm,
        gemmTunable<BM, BN, BK, TM, TN>,
        kBlockThreads,
        0));

    const int activeWarps = activeBlocksPerSm * ((kBlockThreads + 31) / 32);
    const int maximumWarps = properties.maxThreadsPerMultiProcessor / 32;
    registersPerThread = attributes.numRegs;
    staticSharedBytes = attributes.sharedSizeBytes;
    theoreticalOccupancyPct = 100.0 * activeWarps / maximumWarps;
    blockThreads = kBlockThreads;
}

struct Config {
    const char* name;
    int bm;
    int bn;
    int bk;
    int tm;
    int tn;
    LaunchFn launch;
    MetadataFn metadata;
    int registersPerThread = 0;
    size_t staticSharedBytes = 0;
    double theoreticalOccupancyPct = 0.0;
    int blockThreads = 0;
};

#define GEMM_CONFIG(BM, BN, BK, TM, TN)                                     \
    {"bm" #BM "_bn" #BN "_bk" #BK "_tm" #TM "_tn" #TN,                 \
     BM, BN, BK, TM, TN,                                                     \
     &launchKernel<BM, BN, BK, TM, TN>,                                      \
     &queryMetadata<BM, BN, BK, TM, TN>}

std::vector<Config> makeConfigs()
{
    return {
        GEMM_CONFIG(32, 32, 8, 2, 2),
        GEMM_CONFIG(64, 64, 8, 4, 4),
        GEMM_CONFIG(64, 64, 16, 4, 4),
        GEMM_CONFIG(64, 64, 32, 4, 4),
        GEMM_CONFIG(64, 128, 8, 4, 8),
        GEMM_CONFIG(64, 128, 16, 4, 8),
        GEMM_CONFIG(128, 64, 8, 8, 4),
        GEMM_CONFIG(128, 64, 16, 8, 4),
        GEMM_CONFIG(128, 128, 8, 8, 8),
        GEMM_CONFIG(128, 128, 16, 8, 8),
        GEMM_CONFIG(128, 128, 32, 8, 8),
    };
}

struct DeviceBuffers {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    ~DeviceBuffers()
    {
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
    }
};

void initializeInputs(std::vector<float>& A, std::vector<float>& B)
{
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.01f;
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.01f;
    }
}

template <typename Launch>
double benchmarkMilliseconds(Launch launch, int warmups, int repeats)
{
    for (int i = 0; i < warmups; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return totalMs / repeats;
}

void launchCublas(
    cublasHandle_t handle,
    cublasComputeType_t computeType,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS 使用列主序。把 row-major 的 C=A*B 视为 C^T=B^T*A^T。
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_32F,
        N,
        A,
        CUDA_R_32F,
        K,
        &beta,
        C,
        CUDA_R_32F,
        N,
        computeType,
        CUBLAS_GEMM_DEFAULT));
}

double maxAbsoluteError(
    const std::vector<float>& reference,
    const std::vector<float>& actual)
{
    double error = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        error = std::max(
            error,
            std::abs(static_cast<double>(reference[i]) - actual[i]));
    }
    return error;
}

double maxAbsoluteValue(const std::vector<float>& values)
{
    double result = 0.0;
    for (float value : values) {
        result = std::max(result, std::abs(static_cast<double>(value)));
    }
    return result;
}

double gflopsFor(int M, int N, int K, double milliseconds)
{
    const double operations = 2.0 * M * N * K;
    return operations / (milliseconds * 1.0e6);
}

int adaptiveRepeats(int size)
{
    if (size <= 512) return 100;
    if (size <= 1024) return 50;
    if (size <= 2048) return 20;
    return 10;
}

std::vector<int> parseSizes(const std::string& text)
{
    std::vector<int> sizes;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const int size = std::stoi(token);
        if (size <= 0) {
            throw std::runtime_error("matrix size must be positive");
        }
        sizes.push_back(size);
    }
    if (sizes.empty()) {
        throw std::runtime_error("--sizes requires at least one value");
    }
    return sizes;
}

struct Measurement {
    const Config* config = nullptr;
    double milliseconds = 0.0;
    double gflops = 0.0;
    double maxError = 0.0;
    bool passed = false;
};

void writeCsvRow(
    std::ofstream& csv,
    const std::string& device,
    int size,
    const std::string& implementation,
    const std::string& kind,
    const Config* config,
    double milliseconds,
    double gflops,
    double maxError,
    bool passed)
{
    const double problemAi =
        (2.0 * size * size * size) /
        (4.0 * (3.0 * static_cast<double>(size) * size));
    const double tileAi = config
        ? (2.0 * config->bm * config->bn) /
              (4.0 * (config->bm + config->bn))
        : 0.0;

    csv << '"' << device << "\"," << size << ',' << size << ',' << size
        << ',' << implementation << ',' << kind << ','
        << (config ? config->bm : 0) << ','
        << (config ? config->bn : 0) << ','
        << (config ? config->bk : 0) << ','
        << (config ? config->tm : 0) << ','
        << (config ? config->tn : 0) << ','
        << (config ? config->blockThreads : 0) << ','
        << (config ? config->registersPerThread : 0) << ','
        << (config ? config->staticSharedBytes : 0) << ','
        << (config ? config->theoreticalOccupancyPct : 0.0) << ','
        << milliseconds << ',' << gflops << ',' << problemAi << ',' << tileAi
        << ',' << maxError << ',' << (passed ? 1 : 0) << '\n';
}

void runProfileMode(
    const Config& config,
    int size,
    int warmups,
    const cudaDeviceProp& properties)
{
    const size_t elements = static_cast<size_t>(size) * size;
    const size_t bytes = elements * sizeof(float);
    std::vector<float> hostA(elements);
    std::vector<float> hostB(elements);
    initializeInputs(hostA, hostB);

    DeviceBuffers buffers;
    CUDA_CHECK(cudaMalloc(&buffers.A, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.B, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.C, bytes));
    CUDA_CHECK(cudaMemcpy(buffers.A, hostA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.B, hostB.data(), bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmups; ++i) {
        config.launch(
            buffers.A, buffers.B, buffers.C, size, size, size, nullptr);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // scripts/profile_ncu.sh 使用 --launch-skip=warmups，只采集下面这一轮。
    config.launch(buffers.A, buffers.B, buffers.C, size, size, size, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "NCU profile target completed\n"
              << "Device: " << properties.name << '\n'
              << "Config: " << config.name << '\n'
              << "Matrix: " << size << " x " << size << " x " << size << '\n'
              << "Warmups: " << warmups << ", profiled launches: 1\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
    std::string csvPath = "profiles/autotune_results.csv";
    std::string profileConfig;
    int profileSize = 2048;
    int requestedRepeats = 0;
    bool listConfigs = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--sizes" && i + 1 < argc) {
                sizes = parseSizes(argv[++i]);
            } else if (argument == "--csv" && i + 1 < argc) {
                csvPath = argv[++i];
            } else if (argument == "--repeats" && i + 1 < argc) {
                requestedRepeats = std::stoi(argv[++i]);
                if (requestedRepeats <= 0) {
                    throw std::runtime_error("--repeats must be positive");
                }
            } else if (argument == "--profile" && i + 1 < argc) {
                profileConfig = argv[++i];
            } else if (argument == "--size" && i + 1 < argc) {
                profileSize = std::stoi(argv[++i]);
            } else if (argument == "--list-configs") {
                listConfigs = true;
            } else if (argument == "--help") {
                std::cout
                    << "Usage:\n"
                    << "  autotune_gemm [--sizes 256,512,...] [--repeats N] [--csv PATH]\n"
                    << "  autotune_gemm --profile CONFIG [--size N]\n"
                    << "  autotune_gemm --list-configs\n";
                return EXIT_SUCCESS;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + argument);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Argument error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceId));

    std::vector<Config> configs = makeConfigs();
    for (Config& config : configs) {
        config.metadata(
            properties,
            config.registersPerThread,
            config.staticSharedBytes,
            config.theoreticalOccupancyPct,
            config.blockThreads);
    }

    if (listConfigs) {
        std::cout << "Device: " << properties.name << " (SM "
                  << properties.major << '.' << properties.minor << ")\n"
                  << "SM count: " << properties.multiProcessorCount << '\n'
                  << "Memory bus width: " << properties.memoryBusWidth << " bit\n";
        std::cout << "Config, Threads, Registers/thread, Static shared, Theoretical occupancy\n";
        for (const Config& config : configs) {
            std::cout << config.name << ", " << config.blockThreads << ", "
                      << config.registersPerThread << ", "
                      << config.staticSharedBytes << " B, "
                      << std::fixed << std::setprecision(1)
                      << config.theoreticalOccupancyPct << "%\n";
        }
        return EXIT_SUCCESS;
    }

    if (!profileConfig.empty()) {
        const auto found = std::find_if(
            configs.begin(), configs.end(), [&](const Config& config) {
                return profileConfig == config.name;
            });
        if (found == configs.end()) {
            std::cerr << "Unknown config: " << profileConfig
                      << ". Run --list-configs first.\n";
            return EXIT_FAILURE;
        }
        runProfileMode(*found, profileSize, kDefaultWarmups, properties);
        return EXIT_SUCCESS;
    }

    std::filesystem::path csvFile(csvPath);
    if (csvFile.has_parent_path()) {
        std::filesystem::create_directories(csvFile.parent_path());
    }
    std::ofstream csv(csvPath);
    if (!csv) {
        std::cerr << "Cannot open CSV output: " << csvPath << '\n';
        return EXIT_FAILURE;
    }
    csv << std::fixed << std::setprecision(6);
    csv << "device,m,n,k,implementation,kind,bm,bn,bk,tm,tn,threads,"
           "registers_per_thread,static_smem_bytes,theoretical_occupancy_pct,"
           "time_ms,gflops,problem_ai_flop_per_byte,tile_ai_flop_per_byte,"
           "max_abs_error,passed\n";

    std::cout << "============================================================\n"
              << "Parameterized FP32 GEMM Auto-Tuning\n"
              << "Device: " << properties.name << " (SM " << properties.major
              << '.' << properties.minor << ", "
              << properties.multiProcessorCount << " SMs)\n"
              << "Candidates: " << configs.size() << '\n'
              << "CSV: " << csvPath << '\n'
              << "============================================================\n";

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    bool allPassed = true;
    for (int size : sizes) {
        const size_t elements = static_cast<size_t>(size) * size;
        const size_t bytes = elements * sizeof(float);
        std::vector<float> hostA(elements);
        std::vector<float> hostB(elements);
        std::vector<float> reference(elements);
        std::vector<float> actual(elements);
        initializeInputs(hostA, hostB);

        DeviceBuffers buffers;
        CUDA_CHECK(cudaMalloc(&buffers.A, bytes));
        CUDA_CHECK(cudaMalloc(&buffers.B, bytes));
        CUDA_CHECK(cudaMalloc(&buffers.C, bytes));
        CUDA_CHECK(cudaMemcpy(
            buffers.A, hostA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            buffers.B, hostB.data(), bytes, cudaMemcpyHostToDevice));

        const int repeats = requestedRepeats > 0
            ? requestedRepeats
            : adaptiveRepeats(size);

        const auto cublasFp32Launch = [&]() {
            launchCublas(
                handle,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                buffers.A,
                buffers.B,
                buffers.C,
                size,
                size,
                size);
        };
        const double cublasFp32Ms = benchmarkMilliseconds(
            cublasFp32Launch, kDefaultWarmups, repeats);
        CUDA_CHECK(cudaMemcpy(
            reference.data(), buffers.C, bytes, cudaMemcpyDeviceToHost));
        const double referenceMagnitude = maxAbsoluteValue(reference);
        const double tolerance = 1.0e-3 + 1.0e-4 * referenceMagnitude;

        const auto cublasTf32Launch = [&]() {
            launchCublas(
                handle,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                buffers.A,
                buffers.B,
                buffers.C,
                size,
                size,
                size);
        };
        const double cublasTf32Ms = benchmarkMilliseconds(
            cublasTf32Launch, kDefaultWarmups, repeats);
        CUDA_CHECK(cudaMemcpy(
            actual.data(), buffers.C, bytes, cudaMemcpyDeviceToHost));
        const double cublasTf32Error = maxAbsoluteError(reference, actual);
        const bool cublasTf32Passed = cublasTf32Error <= tolerance;

        std::vector<Measurement> measurements;
        measurements.reserve(configs.size());

        for (const Config& config : configs) {
            const auto launch = [&]() {
                config.launch(
                    buffers.A,
                    buffers.B,
                    buffers.C,
                    size,
                    size,
                    size,
                    nullptr);
            };
            const double milliseconds = benchmarkMilliseconds(
                launch, kDefaultWarmups, repeats);
            CUDA_CHECK(cudaMemcpy(
                actual.data(), buffers.C, bytes, cudaMemcpyDeviceToHost));
            const double error = maxAbsoluteError(reference, actual);
            const bool passed = error <= tolerance;
            const double gflops = gflopsFor(size, size, size, milliseconds);
            measurements.push_back(
                {&config, milliseconds, gflops, error, passed});
            writeCsvRow(
                csv,
                properties.name,
                size,
                config.name,
                "custom_fp32",
                &config,
                milliseconds,
                gflops,
                error,
                passed);
            allPassed = allPassed && passed;
        }

        const auto best = std::max_element(
            measurements.begin(), measurements.end(),
            [](const Measurement& left, const Measurement& right) {
                const double leftValue = left.passed ? left.gflops : -1.0;
                const double rightValue = right.passed ? right.gflops : -1.0;
                return leftValue < rightValue;
            });

        const double cublasFp32Gflops = gflopsFor(
            size, size, size, cublasFp32Ms);
        const double cublasTf32Gflops = gflopsFor(
            size, size, size, cublasTf32Ms);
        writeCsvRow(
            csv,
            properties.name,
            size,
            "cuBLAS_FP32_pedantic",
            "cublas_fp32",
            nullptr,
            cublasFp32Ms,
            cublasFp32Gflops,
            0.0,
            true);
        writeCsvRow(
            csv,
            properties.name,
            size,
            "cuBLAS_TF32",
            "cublas_tf32",
            nullptr,
            cublasTf32Ms,
            cublasTf32Gflops,
            cublasTf32Error,
            cublasTf32Passed);

        std::cout << std::fixed << std::setprecision(3)
                  << "\nMatrix " << size << " x " << size << " x " << size
                  << " (repeats=" << repeats << ")\n"
                  << "  Best custom : " << best->config->name << ", "
                  << best->milliseconds << " ms, " << best->gflops
                  << " GFLOPS\n"
                  << "  cuBLAS FP32 : " << cublasFp32Ms << " ms, "
                  << cublasFp32Gflops << " GFLOPS, custom/cuBLAS="
                  << (100.0 * best->gflops / cublasFp32Gflops) << "%\n"
                  << "  cuBLAS TF32 : " << cublasTf32Ms << " ms, "
                  << cublasTf32Gflops << " GFLOPS, max error="
                  << cublasTf32Error << '\n'
                  << std::setprecision(6)
                  << "  Max error   : " << best->maxError
                  << " (tolerance=" << tolerance << ", "
                  << (best->passed ? "PASSED" : "FAILED") << ")\n";
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    csv.close();
    std::cout << "\nAuto-tuning " << (allPassed ? "PASSED" : "FAILED")
              << ". Results written to " << csvPath << '\n';
    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
