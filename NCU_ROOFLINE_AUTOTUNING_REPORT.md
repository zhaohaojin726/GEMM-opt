# GEMM 参数化 Auto-Tuning、Nsight Compute 与 Roofline 分析报告

> 实测日期：2026-09-04  
> 测试设备：NVIDIA GeForce RTX 4060 Laptop GPU（Ada，Compute Capability 8.9）  
> 报告范围：自定义 FP32 GEMM、cuBLAS FP32、cuBLAS TF32、内存带宽、寄存器压力、Occupancy、Warp Stall 与 Roofline

## 1. 结论摘要

本次新增的 Auto-Tuning 框架对 `BM/BN/BK/TM/TN` 进行组合搜索。在 11 个候选参数中：

- 256³ 的最佳配置是 `BM=64, BN=64, BK=32, TM=4, TN=4`，达到 **1.215 TFLOPS**。
- 512³～4096³ 的最佳配置均为 `BM=128, BN=128, BK=32, TM=8, TN=8`。
- 自定义 FP32 的峰值为 1024³ 时的 **4.433 TFLOPS**。
- 2048³ 时，自定义最优实现达到 **4.344 TFLOPS**，为严格 FP32 cuBLAS 的 **56.0%**。
- 4096³ 时，自定义实现为严格 FP32 cuBLAS 的 **63.1%**。
- 所有配置和规模均通过正确性检查，记录到的最大绝对误差为 `0.0`。

最重要的 NCU 结论是：**更高的 Occupancy 并不等于更高的 GEMM 性能**。

- 大 tile 配置使用 158 个寄存器/线程，理论和实测 Occupancy 都只有 **16.67%**，但 NCU 下用时 **4.66 ms**。
- 小 tile 配置只使用 48 个寄存器/线程，实测 Occupancy 达到 **79.73%**，但 NCU 下用时反而是 **6.46 ms**。
- 小 tile 的 L2 流量约为大 tile 的两倍，并出现 **53.06% MIO throttle**；更多 resident warp 没能抵消 shared-memory/MIO 管线压力。
- 两个配置的 DRAM 利用率都不到 4%，2048³ GEMM 相对于显存 Roofline 已处在 compute-bound 区域；瓶颈主要位于片上数据搬运、指令发射和计算管线，而不是显存带宽。

这与 NVIDIA 的建议一致：Occupancy 用于判断延迟隐藏能力，但提高 Occupancy 不一定同比提高性能；寄存器带来的指令级并行和数据复用可能使低 Occupancy 内核更快。参见 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)。

## 2. 实验环境

| 项目 | 实测值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Compute Capability | 8.9 |
| SM 数量 | 24 |
| 显存 | 8188 MiB |
| 显存总线 | 128 bit |
| NVIDIA Driver | 595.84 |
| CUDA 编译器 | CUDA 13.0.88 |
| cuBLAS | `libcublas.so.13` |
| Nsight Compute | 2026.2.1 |
| NCU 采样 SM 时钟 | 1.89 GHz |
| NCU 采样 DRAM 时钟 | 7.99 GHz |
| 当前功耗限制 | 50 W（requested 55 W） |

系统原有 Nsight Compute 2021.3.1 与 CUDA 13/RTX 4060 不兼容。本次将 2026.2.1 安装在：

```text
/media/drink_ice/Data/cuda-tools/nsight-compute-2026.2.1
```

工程中的 `.tools/nsight-compute-current` 是指向该目录的软链接；大型 `.ncu-rep` 文件也保存在 Data 分区，避免占用根分区。

已有的 Nsight Systems 2026.4.1 也已迁移到 `/media/drink_ice/Data/cuda-tools/nsight-systems-2026.4.1`，原路径改为软链接；根分区因此释放约 607 MB。

NVIDIA 官方当前下载页列出了 Ada AD10x 支持范围，见 [Nsight Compute Getting Started](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)。

## 3. Auto-Tuning 框架

### 3.1 参数含义

参数化 kernel 位于 `autotune/autotune_gemm.cu`：

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemmTunable(...)
```

| 参数 | 含义 |
|---|---|
| `BM` | 一个 thread block 计算的 C tile 高度 |
| `BN` | 一个 thread block 计算的 C tile 宽度 |
| `BK` | 每次加载到 shared memory 后沿 K 方向累计的长度 |
| `TM` | 每个线程负责的输出行数，即寄存器 tile 高度 |
| `TN` | 每个线程负责的输出列数，即寄存器 tile 宽度 |

固定实现特征：

- 256 threads/block；
- A、B 的 global-memory 加载使用 `float4`；
- A tile 使用 `BK+1` padding 减少 shared-memory bank conflict；
- 每个线程用 `TM×TN` 个累加寄存器完成 outer product；
- 边界位置自动补零，输出写入带边界检查；
- `cudaFuncGetAttributes` 读取编译后的寄存器和 shared-memory 用量；
- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 计算理论 Occupancy；
- 普通模式搜索全部配置，`--profile` 模式只执行一个配置供 NCU 过滤。

### 3.2 搜索空间和 2048³ 结果

| BM×BN×BK / TM×TN | Reg/thread | Shared/block | 理论 Occ. | 时间 (ms) | GFLOPS |
|---|---:|---:|---:|---:|---:|
| 32×32×8 / 2×2 | 38 | 2.13 KiB | 100.0% | 10.924 | 1572.7 |
| 64×64×8 / 4×4 | 48 | 4.25 KiB | 83.3% | 5.398 | 3182.9 |
| 64×64×16 / 4×4 | 48 | 8.25 KiB | 83.3% | 5.529 | 3107.3 |
| 64×64×32 / 4×4 | 48 | 16.25 KiB | 83.3% | 5.504 | 3121.3 |
| 64×128×8 / 4×8 | 90 | 6.25 KiB | 33.3% | 5.073 | 3386.6 |
| 64×128×16 / 4×8 | 80 | 12.25 KiB | 50.0% | 4.425 | 3882.8 |
| 128×64×8 / 8×4 | 64 | 6.50 KiB | 66.7% | 4.701 | 3654.7 |
| 128×64×16 / 8×4 | 63 | 12.50 KiB | 66.7% | 4.356 | 3944.3 |
| 128×128×8 / 8×8 | 168 | 8.50 KiB | 16.7% | 4.334 | 3964.0 |
| 128×128×16 / 8×8 | 153 | 16.50 KiB | 16.7% | 4.072 | 4219.4 |
| **128×128×32 / 8×8** | **158** | **32.50 KiB** | **16.7%** | **3.955** | **4343.7** |

这个结果说明参数之间不能独立选择：增大 `BK` 不一定更快；只有当 `BM/BN` 提供足够数据复用、`TM/TN` 提供足够寄存器级 ILP 时，较大的 `BK` 才能摊薄同步和循环开销。

### 3.3 测量方法

- 每个配置预热 5 次。
- 计时使用 CUDA Event，只包含 GEMM kernel/cuBLAS 调用。
- 256、512、1024、2048、4096 的重复次数分别为 100、100、50、20、10。
- Host↔Device 拷贝、输入初始化和正确性比较均不进入计时区间。
- 自定义 FP32 以 `CUBLAS_COMPUTE_32F_PEDANTIC` 结果作为正确性参考。
- 另测 `CUBLAS_COMPUTE_32F_FAST_TF32`，用于展示允许 TF32 Tensor Core 后的性能上限，不把它当作严格等精度比较。

根据 [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)，`CUBLAS_COMPUTE_32F_PEDANTIC` 在所有计算阶段使用 32 位单精度；`CUBLAS_COMPUTE_32F_FAST_TF32` 允许把 FP32 输入转换为 TF32 并使用 Tensor Core。

## 4. 多矩阵规模与 cuBLAS 对比

| M=N=K | 最优配置 | 自定义时间 | 自定义 GFLOPS | cuBLAS FP32 GFLOPS | 达到 FP32 cuBLAS | cuBLAS TF32 GFLOPS |
|---:|---|---:|---:|---:|---:|---:|
| 256 | 64×64×32 / 4×4 | 0.0276 ms | 1216.8 | 2899.8 | 42.0% | 4206.4 |
| 512 | 128×128×32 / 8×8 | 0.0781 ms | 3435.6 | 5435.3 | 63.2% | 8126.0 |
| 1024 | 128×128×32 / 8×8 | 0.4844 ms | **4433.2** | 8629.5 | 51.4% | 14029.6 |
| 2048 | 128×128×32 / 8×8 | 3.9551 ms | 4343.7 | 7757.5 | 56.0% | 9509.5 |
| 4096 | 128×128×32 / 8×8 | 38.3690 ms | 3582.0 | 5678.1 | 63.1% | 11577.7 |

分析：

1. 256³ 时，128×128 tile 只产生 4 个 block，无法填满 24 个 SM；64×64 tile 提供更多 block，因此成为最佳配置。
2. 从 512³ 开始，并行 block 数已足够，大 tile 的数据复用开始占主导。
3. cuBLAS 对不同规模选择不同的高度优化 kernel/算法，Auto-Tuner 仍只搜索一个教学型 kernel 的 tile 参数，因此 51%～63% 的严格 FP32 cuBLAS 性能是合理结果。
4. TF32 数据不是等精度对照。它使用 Tensor Core，1024³ 达到 14.0 TFLOPS；本组输入上的最大绝对误差为 `4e-6～1.2e-5`。若目标允许 TF32，后续优化方向应改成 Tensor Core pipeline，而不是继续堆叠 FP32 CUDA Core 微优化。

完整原始数据保存在 `profiles/autotune_results.csv`。

## 5. Roofline 分析

Roofline 的基本关系是：

```text
可达到性能 <= min(计算峰值, 内存带宽 × 算术强度)
算术强度 AI = FLOP / Byte
ridge point = 计算峰值 / 内存带宽
```

NVIDIA 对 Roofline 轴、memory boundary、compute boundary 和 ridge point 的定义见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)。

### 5.1 设备 Roofline 估计

以 NCU 采样时钟计算：

```text
FP32 peak ≈ 24 SM × 128 FP32 lane/SM × 2 FLOP/FMA × 1.89 GHz
          ≈ 11.61 TFLOPS

DRAM peak ≈ 2 × 7.99 GHz × 128 bit / 8
          ≈ 255.68 GB/s

ridge point ≈ 11612 / 255.68
            ≈ 45.4 FLOP/Byte
```

该峰值是采样频率下的估计值；Laptop GPU 会受功耗、温度和动态 boost 影响，因此不应把它当作固定铭牌值。

### 5.2 2048³ 的实测点

一次 2048³ GEMM 完成 `2×2048³ = 17.180 GFLOP`。

| NCU 指标 | 128×128×32 / 8×8 | 64×64×16 / 4×4 |
|---|---:|---:|
| NCU Duration | 4.66 ms | 6.46 ms |
| NCU 推算性能 | 3.69 TFLOPS | 2.66 TFLOPS |
| NCU FP32 peak 百分比 | 32% | 23% |
| DRAM read | 33.69 MB | 33.73 MB |
| DRAM write | 10.28 MB | 10.53 MB |
| DRAM bandwidth | 9.44 GB/s | 6.86 GB/s |
| DRAM peak 利用率 | 3.69% | 2.68% |
| 实测 DRAM AI | 390.7 FLOP/Byte | 388.2 FLOP/Byte |
| L2 traffic | 554.21 MB | 1.09 GB |
| L2 层 AI | 31.0 FLOP/Byte | 约 15.8 FLOP/Byte |
| L2 hit rate | 93.90% | 96.88% |

两个配置的 DRAM AI 都远高于 45.4 FLOP/Byte 的 ridge point，因此相对于显存 Roofline 都属于 **compute-bound**。DRAM 利用率只有约 3%，增加显存带宽不会成为当前主要收益来源。

tile 的理论片上算术强度为：

```text
AI_tile = 2×BM×BN×BK / [4×(BM×BK + BK×BN)]
        = BM×BN / [2×(BM+BN)]
```

- 128×128 tile 的 `AI_tile=32 FLOP/Byte`，与实测 L2 层 AI 31.0 很接近。
- 64×64 tile 的 `AI_tile=16 FLOP/Byte`，与实测约 15.8 很接近。

这解释了大 tile 的优势：它没有显著减少 A、B 的首次 DRAM 读取，却把 L2→L1/shared 的流量从约 1.09 GB 降到 554 MB。

注意：NCU 的 DRAM write 只计算 kernel 执行期间实际到达 DRAM 的流量；部分 C 数据在 kernel 结束时仍可能驻留在 L2，所以实测 DRAM AI 可以略高于按 `A+B+C` 最小流量计算的算法 AI。

## 6. 寄存器压力与 Occupancy

| 指标 | 大 tile 最优配置 | 小 tile 对照配置 |
|---|---:|---:|
| Registers/thread | 158 | 48 |
| Static shared/block | 33.28 KB | 8.45 KB |
| Register block limit/SM | 1 | 5 |
| Shared-memory block limit/SM | 1 | 6 |
| Theoretical active warps/SM | 8 | 40 |
| Theoretical Occupancy | 16.67% | 83.33% |
| Achieved Occupancy | 16.67% | 79.73% |
| Local-memory spilling | 0 | 0 |

大 tile 的 `8×8=64` 个 FP32 累加器是寄存器压力的主要来源。它同时因为 33.28 KB shared memory 只能在每个 SM 驻留一个 block。虽然如此，它没有发生 register spill，并通过更大的寄存器 tile 获得：

- 更高的指令级并行度；
- 更少的 shared-memory 重复读取；
- 更高的片上算术强度；
- 更少的 block 和同步开销。

因此不能仅依据 Occupancy 选择配置。这里 Auto-Tuning 的作用正是让“寄存器复用”和“warp 数量”由实测结果决定。

## 7. Warp Scheduler 与 Stall

### 7.1 Scheduler 状态

| 指标 | 大 tile 最优配置 | 小 tile 对照配置 |
|---|---:|---:|
| Active warps/scheduler | 2.00 | 9.57 |
| Eligible warps/scheduler | 0.72 | 1.94 |
| Issued warp/scheduler | 0.46 | 0.42 |
| No eligible cycles | 53.52% | 57.56% |
| Warp cycles/issued instruction | 4.30 | 22.54 |

小 tile 虽然有更多 active/eligible warp，但每周期实际发射率反而略低。原因可从 Stall 分布看到。

### 7.2 Stall 原因

| Warp Stall | 大 tile 最优配置 | 小 tile 对照配置 | 含义 |
|---|---:|---:|---|
| MIO throttle | 24.84% | **53.06%** | shared-memory/MIO 指令队列压力 |
| Long scoreboard | **20.72%** | 6.14% | 等待 L1TEX/global/local 数据依赖 |
| Not selected | 12.53% | 15.86% | warp 已可执行，但 scheduler 选择了其他 warp |
| Wait | 8.35% | 5.26% | 等待固定延迟执行依赖 |
| Barrier | 1.93% | 7.20% | 等待 block 内同步 |
| Short scoreboard | 0.95% | 4.76% | 等待 shared-memory/MIO 短依赖 |
| Math pipe throttle | 0.39% | 0.17% | 数学执行管线繁忙 |
| LG throttle | 0.09% | 0.00% | global/local load-store 队列压力 |

小 tile 的核心问题是 MIO throttle：它的 L2 流量约为大 tile 的两倍，并需要更多 shared-memory 指令和同步。NCU 规则也明确指出，53.1% 的周期位于 MIO queue stall。

大 tile 将 MIO throttle 降到 24.84%，但由于只有两个 active warp/scheduler，20.72% long-scoreboard 更难被其他 warp 隐藏。下一步优化应围绕 `cp.async`/双缓冲和 shared-memory 访问减少，而不是简单使用 `-maxrregcount` 强制压低寄存器；后者很可能导致 spill，破坏当前优势。

NVIDIA 建议只有当 scheduler 无法持续发射时才重点分析 Stall，并强调 Stall 并不一定都可避免。各状态定义见 [Nsight Compute Warp State Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-scheduler-states)。

## 8. 后续优化建议

按预期收益排序：

1. **为大 tile 加入 `cp.async` 双缓冲**：目标是把下一块 A/B 的 global→shared 加载与当前 BK 计算重叠，降低 long-scoreboard。
2. **减少 shared-memory/MIO 指令数**：尝试 shared-memory vector load，或在 warp 内用 shuffle 广播 A/B；重点观察 MIO throttle 是否下降。
3. **扩大寄存器压力附近的搜索空间**：增加 `TM×TN=8×4、4×8、8×6、6×8` 等配置，寻找 64～120 registers/thread 的中间点。
4. **按矩阵形状调参**：当前只搜索方阵。实际框架可扩展到独立 `M/N/K`，为 tall-skinny、wide-short 和小 K 建立不同候选集合。
5. **若允许 TF32/FP16，改用 Tensor Core pipeline**：cuBLAS TF32 已表明硬件仍有明显 Tensor Core 性能空间。
6. **控制功耗和温度后复测**：Laptop GPU 的 50～55 W 功耗限制会让不同测试批次的时钟变化，正式比较建议固定电源模式并记录温度、频率。

## 9. 复现方法

### 9.1 编译和 Auto-Tuning

```bash
make autotune

./bin/autotune_gemm \
    --sizes 256,512,1024,2048,4096 \
    --csv profiles/autotune_results.csv
```

查看所有候选的静态资源信息：

```bash
./bin/autotune_gemm --list-configs
```

### 9.2 采集 NCU

```bash
./scripts/profile_ncu.sh bm128_bn128_bk32_tm8_tn8 2048
./scripts/profile_ncu.sh bm64_bn64_bk16_tm4_tn4 2048
```

脚本使用：

- `--kernel-name regex:gemmTunable`
- `--launch-skip 5`
- `--launch-count 1`
- Roofline、SpeedOfLight、MemoryWorkloadAnalysis、LaunchStats、Occupancy、SchedulerStats、WarpStateStats sections
- DRAM/L2 字节计数和九类 Warp Stall 指标

NCU 为本次完整 section 集合执行了 17 次 kernel replay。官方文档说明，多指标采集可能需要 replay 并引入显著 profiling overhead，因此本报告使用普通 CUDA Event 结果比较 GFLOPS，只使用 NCU 运行解释硬件行为。参见 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 和 [Metric Collection Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead)。

打开原始报告：

```bash
.tools/nsight-compute-current/ncu-ui profiles/ncu/<report>.ncu-rep
```

或在命令行查看：

```bash
.tools/nsight-compute-current/ncu \
    --import profiles/ncu/<report>.ncu-rep \
    --page details
```

本次分析使用的两份最终报告为：

```text
profiles/ncu/autotune_bm128_bn128_bk32_tm8_tn8_2048_20260904_184146.ncu-rep
profiles/ncu/autotune_bm64_bn64_bk16_tm4_tn4_2048_20260904_184509.ncu-rep
```

## 10. 文件清单

| 文件 | 作用 |
|---|---|
| `autotune/autotune_gemm.cu` | 参数化 kernel、搜索器、正确性检查与 cuBLAS 对照 |
| `scripts/profile_ncu.sh` | 固定 kernel 过滤和指标集合的 NCU 采集脚本 |
| `profiles/autotune_results.csv` | 五种矩阵规模的完整原始计时数据 |
| `profiles/ncu/` | 指向 Data 分区 NCU 报告目录的软链接 |
| `NCU_ROOFLINE_AUTOTUNING_REPORT.md` | 本报告 |
