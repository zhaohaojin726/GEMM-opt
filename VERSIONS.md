# GEMM 独立版本

每个优化版本现在都是一个可以单独编译、运行和分析的 CUDA 程序：

| 文件 | 版本 | 主要优化 |
|---|---|---|
| `versions/v0_naive.cu` | V0 | 每线程计算一个输出元素 |
| `versions/v1_shared_memory.cu` | V1 | Shared-memory tiling |
| `versions/v2_register_tiling.cu` | V2 | Register tiling |
| `versions/v3_vectorized_load.cu` | V3 | `float4` 向量化加载 |
| `versions/v4_double_buffer.cu` | V4 | 寄存器预取与 shared-memory 双缓冲 |
| `versions/v5_cp_async.cu` | V5 | `cp.async` 与 shared-memory 双缓冲 |
| `versions/v6_warp_tiling.cu` | V6 | Warp tiling |
| `versions/v7_tensor_core.cu` | V7 | FP16/FP32 WMMA Tensor Core |

公共的矩阵初始化、CUDA Event 计时、FP32 reference 和误差检查位于
`include/gemm_benchmark.cuh`。原来的 `gemm.cu` 继续作为一次运行全部版本的横向对比程序。

## 编译

编译所有独立版本和横向对比版本：

```bash
make
```

只编译某个版本：

```bash
make bin/v6_warp_tiling
make bin/v7_tensor_core
```

直接使用 NVCC：

```bash
nvcc -O3 -std=c++17 -lineinfo \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_80,code=compute_80 \
  versions/v6_warp_tiling.cu -o bin/v6_warp_tiling
```

如果 Python 版 CUDA 13 报错找不到 `nv/target`，需要补齐 NVIDIA CCCL 头文件：

```bash
python -m pip install nvidia-cuda-cccl
```

## 运行

```bash
./bin/v0_naive
./bin/v6_warp_tiling
./bin/v7_tensor_core
```

每个程序只计时自己的优化 kernel，但都会额外执行一次普通 FP32 reference kernel
进行正确性检查。Reference kernel 不计入输出的执行时间。

## Nsight Compute

独立程序不再需要 `--launch-skip 5`，可以直接按照 kernel 名称分析：

```bash
ncu \
  --section-folder /usr/lib/nsight-compute/sections \
  --kernel-name-base function \
  --kernel-name gemmWarpTiled \
  --launch-count 1 \
  --set default \
  ./bin/v6_warp_tiling
```

Tensor Core 版本把 kernel 名改为 `gemmTensorCore`，程序改为
`./bin/v7_tensor_core` 即可。

## Nsight Systems

项目已提供用户级 Nsight Systems 配置脚本。默认关闭需要额外 Linux
`perf_event` 权限的 CPU sampling，只采集 CUDA、NVTX 和 OS runtime 时间线：

```bash
make profile-v6
make profile-v7
```

也可以分析任意独立版本：

```bash
./scripts/profile_nsys.sh v4_double_buffer
```

报告保存在 `profiles/nsys/`，文件名包含版本和采集时间。查看命令行统计：

```bash
nsys stats profiles/nsys/<report>.nsys-rep
```
