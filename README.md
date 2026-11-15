
# CUDA GEMM Optimization

A CUDA-based GEMM (General Matrix Multiplication) optimization project.

## Overview

This project implements and compares multiple CUDA GEMM kernels, progressively exploring memory hierarchy, data reuse, instruction-level parallelism, asynchronous execution, and Tensor Core acceleration.

## Optimization Roadmap

| Version | Optimization | Key Idea |
|---|---|---|
| V0 | Naive GEMM | One output element per thread |
| V1 | Shared Memory Tiling | Cooperative loading and data reuse |
| V2 | Register Tiling | Multiple accumulators per thread |
| V3 | Vectorized Load | Wider memory load operations |
| V4 | Double Buffering | Overlap memory staging and computation |
| V5 | cp.async | Asynchronous Global-to-Shared copies |
| V6 | Warp Tiling | Warp-level output tile mapping |
| V7 | Tensor Core | Hardware matrix-multiply acceleration |

## Hardware

- GPU: NVIDIA RTX 4060 Laptop GPU
- Compute Capability: SM 8.9
- CUDA Toolkit: 13.0

## Build

```bash
make GENCODE_FLAGS="-gencode arch=compute_89,code=sm_89"
```

## Benchmark

Benchmark results will be reported with matrix shapes, numerical precision, execution times, correctness tolerances, and hardware configuration.

FP32 CUDA Core implementations and FP16-input Tensor Core implementations should be compared with their precision differences explicitly noted.

## Optimization Principles

- Global Memory Coalescing
- Shared Memory Reuse
- Register Tiling
- Warp-level Computation Mapping
- Memory/Compute Overlap
- Tensor Core Acceleration

