NVCC ?= nvcc

NVCC_FLAGS ?= -O3 -std=c++17 -lineinfo
GENCODE_FLAGS ?= \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_80,code=compute_80
CPPFLAGS ?=

VERSIONS := \
	v0_naive \
	v1_shared_memory \
	v2_register_tiling \
	v3_vectorized_load \
	v4_double_buffer \
	v5_cp_async \
	v6_warp_tiling \
	v7_tensor_core

VERSION_BINARIES := $(addprefix bin/,$(VERSIONS))

.PHONY: all versions combined autotune profile-v6 profile-v7 profile-ncu

all: versions combined

versions: $(VERSION_BINARIES)

combined: bin/gemm_all

autotune: bin/autotune_gemm

bin/%: versions/%.cu include/gemm_benchmark.cuh
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(CPPFLAGS) $(GENCODE_FLAGS) $< -o $@

bin/gemm_all: gemm.cu
	@mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(CPPFLAGS) $(GENCODE_FLAGS) $< -o $@

CUDA_ROOT ?= $(abspath $(dir $(shell command -v $(NVCC)))/..)
CUBLAS_INCLUDE ?= $(CUDA_ROOT)/include
CUBLAS_LIBRARY ?= $(CUDA_ROOT)/lib
CUBLAS_LINK ?= -l:libcublas.so.13
CCCL_INCLUDE ?= /media/drink_ice/Data/cuda-tools/cuda-cccl/nvidia/cu13/include
AUTOTUNE_GENCODE ?= -gencode arch=compute_89,code=sm_89

bin/autotune_gemm: autotune/autotune_gemm.cu
	@mkdir -p bin profiles
	$(NVCC) $(NVCC_FLAGS) $(CPPFLAGS) $(AUTOTUNE_GENCODE) \
		-I$(CCCL_INCLUDE) -I$(CUBLAS_INCLUDE) $< -L$(CUBLAS_LIBRARY) \
		-Xlinker -rpath -Xlinker $(CUBLAS_LIBRARY) $(CUBLAS_LINK) -o $@

profile-v6: bin/v6_warp_tiling
	./scripts/profile_nsys.sh v6_warp_tiling

profile-v7: bin/v7_tensor_core
	./scripts/profile_nsys.sh v7_tensor_core

profile-ncu: bin/autotune_gemm
	./scripts/profile_ncu.sh
