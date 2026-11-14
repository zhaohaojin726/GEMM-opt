# V5 `cp.async` 双缓冲 GEMM 逐行说明

本文对应 `gemm.cu` 中的 `gemmCpAsyncDoubleBuffered`。V5 在 V3 的向量化加载和 V2 的寄存器分块基础上，使用 Ampere（`sm_80`）开始支持的 `cp.async` 指令，把下一块矩阵数据直接从 global memory 异步复制到备用 shared-memory buffer。

## 1. V5 使用的参数

V5 复用前面版本定义的常量：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `TILE_SIZE` | 16 | thread block 的 X、Y 方向线程数 |
| `REGISTER_BLOCK_M` | 64 | 一个 block 负责的 C tile 行数 |
| `REGISTER_BLOCK_N` | 64 | 一个 block 负责的 C tile 列数 |
| `REGISTER_BLOCK_K` | 16 | 每轮处理的 K 长度，即 BK |
| `REGISTER_TILE_M` | 4 | 每线程负责的输出行数 |
| `REGISTER_TILE_N` | 4 | 每线程负责的输出列数 |
| `VECTOR_WIDTH` | 4 | 一条向量复制包含 4 个 float |
| `BUFFER_COUNT` | 2 | shared-memory ping-pong buffer 数量 |

因此：

```text
线程块：16 × 16 = 256 个线程
Block tile：64 × 64 × 16
Thread tile：4 × 4
每线程输出：16 个 C 元素
```

每一轮 K tile 需要的数据是：

```text
A tile：64 × 16 = 1024 float = 4096 Byte
B tile：16 × 64 = 1024 float = 4096 Byte

总计：8192 Byte
```

每个 block 有 256 个线程。每个线程为 A、B 各发出一条 16 Byte 的 `cp.async`：

```text
每线程：16 Byte A + 16 Byte B = 32 Byte
256 × 32 Byte = 8192 Byte
```

刚好覆盖一个完整的 A/B K tile。

## 2. 总体执行流程

```text
Prologue：
    cp.async(tile 0 → buffer 0)
    等待 tile 0

主循环第 0 轮：
    cp.async(tile 1 → buffer 1)  ─┐
                                  ├─ 时间上重叠
    compute(tile 0 in buffer 0)  ─┘
    等待 tile 1
    currentBuffer = 1

主循环第 1 轮：
    cp.async(tile 2 → buffer 0)  ─┐
                                  ├─ 时间上重叠
    compute(tile 1 in buffer 1)  ─┘
    等待 tile 2
    currentBuffer = 0
```

两套 shared buffer 的状态交替变化：

```text
轮次       buffer 0              buffer 1
------------------------------------------------
0          计算 tile 0           异步接收 tile 1
1          异步接收 tile 2       计算 tile 1
2          计算 tile 2           异步接收 tile 3
```

## 3. 异步复制函数

函数声明：

```cpp
__device__ __forceinline__ void copyFloat4GlobalToSharedAsync(
    float* destination,
    const float* matrix,
    int row,
    int col,
    int rows,
    int cols)
```

逐行含义：

```cpp
__device__
```

表示函数只能在 GPU device 代码中调用。

```cpp
__forceinline__
```

要求编译器尽量把函数体直接展开到调用位置，避免 device 函数调用开销，也方便编译器调度 `cp.async`。

```cpp
float* destination
```

目标 shared-memory 地址。V5 会把数据直接复制到这里。

```cpp
const float* matrix
```

源 global-memory 矩阵指针，即 A 或 B。

```cpp
int row, int col
```

当前线程负责复制的向量在源矩阵中的起始行、列。

```cpp
int rows, int cols
```

矩阵的实际行数、列数，用来进行边界检查。

### 3.1 架构条件

```cpp
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
```

只有编译目标至少是 `sm_80` 时，才编译真正的 `cp.async` 路径。

```text
sm_75：__CUDA_ARCH__ = 750，不支持 cp.async
sm_80：__CUDA_ARCH__ = 800，支持 cp.async
sm_86：__CUDA_ARCH__ = 860，支持 cp.async
```

### 3.2 判断是否可以复制 16 Byte

```cpp
bool canVectorize =
    row < rows
    && (cols % VECTOR_WIDTH) == 0
    && (col % VECTOR_WIDTH) == 0
    && col + VECTOR_WIDTH <= cols;
```

逐个条件解释：

```cpp
row < rows
```

当前行必须在矩阵内部。

```cpp
(cols % VECTOR_WIDTH) == 0
```

一行长度必须是 4 个 float 的整数倍。这样下一行的起始地址仍然满足 16 Byte 对齐。

```cpp
(col % VECTOR_WIDTH) == 0
```

起始列必须是 4 的倍数，即从 `0、4、8、12...` 开始复制。

```cpp
col + VECTOR_WIDTH <= cols
```

从 `col` 开始的 4 个 float 必须全部处于当前行内，不能跨越行尾。

### 3.3 计算 global-memory 地址

```cpp
size_t offset =
    static_cast<size_t>(row) * cols
    + col;
```

矩阵采用 row-major 布局，二维坐标转一维下标：

```text
offset = row × 每行元素数 + col
```

`row` 转成 `size_t` 是为了让乘法使用足够宽的无符号整数，降低大型矩阵下标发生 32 位溢出的风险。

```cpp
const float* source = matrix + offset;
```

`source` 指向要复制的第一个 float。后面连续 16 Byte 是 4 个 float。

### 3.4 转换 shared-memory 地址

```cpp
unsigned int sharedAddress =
    static_cast<unsigned int>(
        __cvta_generic_to_shared(destination)
    );
```

CUDA C++ 中的 `destination` 是 generic pointer，而 PTX 的 `cp.async` 目标操作数需要 shared address space 中的地址。

```cpp
__cvta_generic_to_shared(destination)
```

把 generic pointer 转换成 shared-memory 地址。

```cpp
static_cast<unsigned int>(...)
```

shared-memory 地址在 PTX 中使用 32 位地址，因此转换成 `unsigned int`。

### 3.5 发出 `cp.async`

```cpp
asm volatile(
    "cp.async.cg.shared.global [%0], [%1], 16;\n"
    :
    : "r"(sharedAddress), "l"(source)
);
```

这是内联 PTX。指令可以拆成：

```text
cp.async  .cg  .shared  .global  [dst], [src], 16
```

| 部分 | 含义 |
|---|---|
| `cp.async` | 发出异步复制 |
| `.cg` | global cache 策略，数据经过 L2 |
| `.shared` | 目标地址空间是 shared memory |
| `.global` | 源地址空间是 global memory |
| `16` | 复制 16 Byte，即一个 `float4` |

占位符对应关系：

```text
%0 → sharedAddress
%1 → source
```

约束：

```cpp
"r"(sharedAddress)
```

要求 shared 地址放进 32 位通用寄存器。

```cpp
"l"(source)
```

要求 global pointer 放进 64 位地址寄存器。

```cpp
asm volatile
```

告诉编译器这条汇编有重要副作用，不能随意删除。

```cpp
return;
```

成功发出 `cp.async` 后直接返回。注意，这里的返回只代表“复制请求已经发出”，并不代表数据已经到达 shared memory。

### 3.6 同步回退路径

```cpp
float4 value =
    loadFloat4OrZero(
        matrix,
        row,
        col,
        rows,
        cols
    );
```

下列情况会走回退路径：

```text
编译架构低于 sm_80
矩阵尾块越界
行步长不满足 float4 对齐
起始列不满足 float4 对齐
```

`loadFloat4OrZero` 安全读取最多 4 个 float，越界部分补 0。

```cpp
destination[0] = value.x;
destination[1] = value.y;
destination[2] = value.z;
destination[3] = value.w;
```

把读取结果同步写入 shared memory。这个路径保证任意 M、N、K 尺寸下仍然正确，但不提供真正的异步复制。

## 4. 提交和等待异步复制

### 4.1 `commitAsyncCopies`

```cpp
__device__ __forceinline__ void commitAsyncCopies()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : :);
#endif
}
```

```cpp
cp.async.commit_group
```

把当前线程此前尚未提交的 `cp.async` 指令组成一个 group。在 V5 中，每线程通常先发出两条复制：

```text
cp.async A 的 16 Byte
cp.async B 的 16 Byte
commit_group
```

`commit_group` 不是等待操作。执行后，复制仍然可以在后台进行。

### 4.2 `waitForAsyncCopies`

```cpp
__device__ __forceinline__ void waitForAsyncCopies()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" : :);
#endif
}
```

```cpp
cp.async.wait_group 0
```

等待当前线程所有已提交的异步复制 group 完成。参数 0 表示不允许留下尚未完成的旧 group。

它只解决“当前线程发出的异步复制是否完成”，不能替代整个 thread block 的同步，所以后面还需要：

```cpp
__syncthreads();
```

两者分工：

```text
wait_group 0：保证本线程的 cp.async 完成
__syncthreads：保证 block 所有线程到达，并让所有线程安全读取完整 tile
```

## 5. V5 Kernel 声明

```cpp
__global__ void gemmCpAsyncDoubleBuffered(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
```

含义与前面版本一致：

```text
A：M × K
B：K × N
C：M × N
```

## 6. 两套 shared-memory buffer

```cpp
__shared__ __align__(16) float tileA
    [BUFFER_COUNT][REGISTER_BLOCK_M][REGISTER_BLOCK_K];

__shared__ __align__(16) float tileB
    [BUFFER_COUNT][REGISTER_BLOCK_K][REGISTER_BLOCK_N];
```

展开当前常量后：

```cpp
tileA[2][64][16];
tileB[2][16][64];
```

内存占用：

```text
tileA：2 × 64 × 16 × 4 Byte = 8192 Byte
tileB：2 × 16 × 64 × 4 Byte = 8192 Byte
总计：16384 Byte
```

```cpp
__align__(16)
```

保证 shared 数组起始地址为 16 Byte 对齐。每行长度也都是 16 Byte 的整数倍，因此每个 `float4` 目标地址都可以保持对齐。

V5 的 `tileA` 没有像 V2/V4 那样在最后一维补 1，因为 `cp.async` 的 16 Byte 目标地址对齐更重要；当前线程映射下，长度为 16 的 A 行布局仍可正常工作。

## 7. 当前 block 负责的 C tile

```cpp
int blockRow =
    blockIdx.y * REGISTER_BLOCK_M;
```

当前 block 负责的 C tile 起始行。`REGISTER_BLOCK_M=64`，所以 Y 方向相邻 block 相差 64 行。

```cpp
int blockCol =
    blockIdx.x * REGISTER_BLOCK_N;
```

当前 block 负责的 C tile 起始列。`REGISTER_BLOCK_N=64`，所以 X 方向相邻 block 相差 64 列。

## 8. 把二维线程编号变成一维编号

```cpp
int linearThread =
    threadIdx.y * TILE_SIZE
    + threadIdx.x;
```

block 是 `16×16`：

```text
threadIdx.y = 0 → linearThread 0～15
threadIdx.y = 1 → linearThread 16～31
...
threadIdx.y = 15 → linearThread 240～255
```

## 9. A、B 的 float4 加载映射

### 9.1 A tile

```cpp
int aVectorRow =
    linearThread
    / (REGISTER_BLOCK_K / VECTOR_WIDTH);
```

A tile 每行有 16 个 float，即 4 个 `float4`：

```text
REGISTER_BLOCK_K / VECTOR_WIDTH = 16 / 4 = 4
```

所以每 4 个线程负责 A tile 的一行：

```text
线程 0～3   → A tile 第 0 行
线程 4～7   → A tile 第 1 行
...
线程 252～255 → A tile 第 63 行
```

```cpp
int aVectorCol =
    (linearThread % (REGISTER_BLOCK_K / VECTOR_WIDTH))
    * VECTOR_WIDTH;
```

余数决定当前线程负责一行中的第几个 `float4`：

```text
余数 0 → 列 0～3
余数 1 → 列 4～7
余数 2 → 列 8～11
余数 3 → 列 12～15
```

### 9.2 B tile

```cpp
int bVectorRow =
    linearThread
    / (REGISTER_BLOCK_N / VECTOR_WIDTH);
```

B tile 每行有 64 个 float，即 16 个 `float4`：

```text
REGISTER_BLOCK_N / VECTOR_WIDTH = 64 / 4 = 16
```

所以每 16 个线程负责 B tile 的一行：

```text
线程 0～15   → B tile 第 0 行
线程 16～31  → B tile 第 1 行
...
线程 240～255 → B tile 第 15 行
```

```cpp
int bVectorCol =
    (linearThread % (REGISTER_BLOCK_N / VECTOR_WIDTH))
    * VECTOR_WIDTH;
```

余数决定起始列：`0、4、8...60`。

## 10. 每线程的寄存器累加器

```cpp
float accum[REGISTER_TILE_M][REGISTER_TILE_N];
```

展开后是：

```cpp
float accum[4][4];
```

每线程负责 16 个 C 元素，每个元素使用一个寄存器累加。

```cpp
#pragma unroll
for (int i = 0; i < REGISTER_TILE_M; ++i) {
    #pragma unroll
    for (int j = 0; j < REGISTER_TILE_N; ++j) {
        accum[i][j] = 0.0f;
    }
}
```

把 16 个累加器清零。`#pragma unroll` 鼓励编译器完全展开循环，使数组元素标量化到寄存器。

## 11. Prologue：准备第一个 tile

流水线开始时还没有可计算的 tile，因此必须先准备 buffer 0。

### 11.1 复制 A tile 0

```cpp
copyFloat4GlobalToSharedAsync(
    &tileA[0][aVectorRow][aVectorCol],
    A,
    blockRow + aVectorRow,
    aVectorCol,
    M,
    K
);
```

参数逐行解释：

```cpp
&tileA[0][aVectorRow][aVectorCol]
```

复制目标是 shared buffer 0 中当前线程负责的 A `float4`。

```cpp
A
```

复制源矩阵是 global-memory A。

```cpp
blockRow + aVectorRow
```

A 的 global 行号。当前 block 负责 64 行，`aVectorRow` 选择其中一行。

```cpp
aVectorCol
```

第一个 K tile 从 K=0 开始，所以 global 列号就是 `aVectorCol`。

```cpp
M, K
```

A 的实际形状，用于边界判断。

### 11.2 复制 B tile 0

```cpp
copyFloat4GlobalToSharedAsync(
    &tileB[0][bVectorRow][bVectorCol],
    B,
    bVectorRow,
    blockCol + bVectorCol,
    K,
    N
);
```

```cpp
&tileB[0][bVectorRow][bVectorCol]
```

目标是 shared buffer 0 中当前线程负责的 B `float4`。

```cpp
bVectorRow
```

第一个 K tile 从 K=0 开始，所以 B 的 global 行号就是 `bVectorRow`。

```cpp
blockCol + bVectorCol
```

B 的 global 列号。`blockCol` 定位当前 C tile，`bVectorCol` 定位 tile 内列。

```cpp
K, N
```

B 的实际形状。

### 11.3 提交、等待和 block 同步

```cpp
commitAsyncCopies();
waitForAsyncCopies();
__syncthreads();
```

执行顺序：

```text
提交本线程的 A/B cp.async group
等待本线程的 group 完成
等待整个 block 的 256 个线程都完成
```

Prologue 必须等待，因为还没有其他 tile 可以计算。

```cpp
int currentBuffer = 0;
```

第一个可计算 tile 位于 buffer 0。

## 12. K tile 主循环

```cpp
for (int tileK = 0;
     tileK < K;
     tileK += REGISTER_BLOCK_K)
```

`REGISTER_BLOCK_K=16`，因此：

```text
tileK = 0、16、32、48...
```

如果 `K=1024`，循环 64 次。

### 12.1 下一 tile 和下一 buffer

```cpp
int nextTileK = tileK + REGISTER_BLOCK_K;
```

计算下一 K tile 的起始位置。

```cpp
int nextBuffer = currentBuffer ^ 1;
```

在 0、1 之间切换：

```text
0 ^ 1 = 1
1 ^ 1 = 0
```

```cpp
bool hasNextTile = nextTileK < K;
```

判断是否还有下一块。最后一轮没有下一块，不需要继续发出异步复制。

### 12.2 异步复制下一块 A

```cpp
copyFloat4GlobalToSharedAsync(
    &tileA[nextBuffer][aVectorRow][aVectorCol],
    A,
    blockRow + aVectorRow,
    nextTileK + aVectorCol,
    M,
    K
);
```

与 Prologue 的差别：

```text
目标 buffer：0 → nextBuffer
global K 列：aVectorCol → nextTileK + aVectorCol
```

### 12.3 异步复制下一块 B

```cpp
copyFloat4GlobalToSharedAsync(
    &tileB[nextBuffer][bVectorRow][bVectorCol],
    B,
    nextTileK + bVectorRow,
    blockCol + bVectorCol,
    K,
    N
);
```

B 的 K 维在行方向，因此下一块的 global 行号是：

```text
nextTileK + bVectorRow
```

```cpp
commitAsyncCopies();
```

把当前线程的 A/B 两条 `cp.async` 组成一个新 group。这里不等待，紧接着开始计算 current buffer。

## 13. 计算当前 buffer

```cpp
#pragma unroll
for (int k = 0; k < REGISTER_BLOCK_K; ++k)
```

遍历当前 K tile 的 16 个 k。

```cpp
float regA[REGISTER_TILE_M];
float regB[REGISTER_TILE_N];
```

展开后：

```cpp
float regA[4];
float regB[4];
```

它们保存当前一个 k 上的 4 个 A 和 4 个 B。

### 13.1 shared A → 寄存器

```cpp
regA[i] =
    tileA[currentBuffer]
         [i * TILE_SIZE + threadIdx.y][k];
```

每线程选择 4 行：

```text
threadIdx.y + 0
threadIdx.y + 16
threadIdx.y + 32
threadIdx.y + 48
```

当前固定为同一个 k。

### 13.2 shared B → 寄存器

```cpp
regB[j] =
    tileB[currentBuffer]
         [k][j * TILE_SIZE + threadIdx.x];
```

每线程选择 4 列：

```text
threadIdx.x + 0
threadIdx.x + 16
threadIdx.x + 32
threadIdx.x + 48
```

### 13.3 `4×4` 外积

```cpp
accum[i][j] += regA[i] * regB[j];
```

4 个 A 与 4 个 B 组合成 16 次 FMA：

```text
             regB0   regB1   regB2   regB3
regA0        acc00    acc01    acc02    acc03
regA1        acc10    acc11    acc12    acc13
regA2        acc20    acc21    acc22    acc23
regA3        acc30    acc31    acc32    acc33
```

每个 k 做 16 次 FMA，BK=16，所以每线程每个 K tile 做：

```text
16 × 16 = 256 次 FMA
```

这一整段计算只读取 `currentBuffer`，而之前的 `cp.async` 正在写 `nextBuffer`：

```text
CUDA Core：读取 currentBuffer，执行 FMA
cp.async：写入 nextBuffer
```

这就是 V5 的计算/访存重叠窗口。

## 14. 等待下一 tile 并切换 buffer

```cpp
if (hasNextTile) {
    waitForAsyncCopies();
    __syncthreads();
    currentBuffer = nextBuffer;
}
```

```cpp
waitForAsyncCopies();
```

如果异步复制在计算期间已经完成，这里几乎不需要等待；否则等待剩余的复制时间。

```cpp
__syncthreads();
```

确保 block 中所有线程负责的 A/B 向量都已经到达 shared memory，并确保所有线程都完成当前 tile 的计算。

```cpp
currentBuffer = nextBuffer;
```

下一轮开始读取刚刚填好的 buffer。

## 15. Epilogue：写回 C

```cpp
for (int i = 0; i < REGISTER_TILE_M; ++i) {
    int row =
        blockRow
        + i * TILE_SIZE
        + threadIdx.y;
```

计算当前线程第 i 个输出行的 global 行号。

```cpp
for (int j = 0; j < REGISTER_TILE_N; ++j) {
    int col =
        blockCol
        + j * TILE_SIZE
        + threadIdx.x;
```

计算当前线程第 j 个输出列的 global 列号。

```cpp
if (row < M && col < N) {
    C[row * N + col] = accum[i][j];
}
```

边界合法时，把寄存器累加结果写回 global-memory C。每线程最多写回 16 个元素。

## 16. 为什么 V5 是真正的异步复制

V4 的路径：

```text
Global → 普通寄存器 nextA/nextB → Shared
```

V4 依靠普通 global load 与独立计算之间的指令级并行隐藏延迟。

V5 的路径：

```text
Global ──cp.async──→ Shared
```

V5 在代码中的关键顺序是：

```cpp
copyFloat4GlobalToSharedAsync(nextBuffer, ...);
commitAsyncCopies();

// 不等待，立即计算 currentBuffer
compute(currentBuffer);

// 到切换点才等待
waitForAsyncCopies();
__syncthreads();
currentBuffer = nextBuffer;
```

编译后的 PTX 顺序已经验证为：

```text
cp.async.cg.shared.global
cp.async.commit_group
当前 tile 的 shared load 和 FMA
cp.async.wait_group 0
bar.sync
```

## 17. 架构检测

主程序先查询设备和 kernel 属性：

```cpp
int device = 0;
cudaDeviceProp deviceProperties{};
cudaFuncAttributes cpAsyncAttributes{};

CUDA_CHECK(cudaGetDevice(&device));
CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));
CUDA_CHECK(
    cudaFuncGetAttributes(
        &cpAsyncAttributes,
        gemmCpAsyncDoubleBuffered
    )
);
```

```cpp
deviceProperties.major
```

表示实际 GPU 的 compute capability 主版本。

```cpp
cpAsyncAttributes.binaryVersion
```

表示当前选中的 kernel binary 是为哪个架构生成的。

```cpp
bool cpAsyncAvailable =
    deviceProperties.major >= 8
    && cpAsyncAttributes.binaryVersion >= 80;
```

两个条件都满足才运行 V5：

```text
GPU 硬件至少为 sm_80
kernel 本身也确实编译出了 sm_80+ 路径
```

这样可以避免“在 Ampere GPU 上误用了只为 sm_75 编译的同步回退 kernel”。

## 18. Benchmark

```cpp
dim3 block(TILE_SIZE, TILE_SIZE);
```

启动 `16×16=256` 个线程。

```cpp
dim3 grid(
    (N + REGISTER_BLOCK_N - 1) / REGISTER_BLOCK_N,
    (M + REGISTER_BLOCK_M - 1) / REGISTER_BLOCK_M
);
```

每个 block 负责 `64×64` 的 C tile，向上取整覆盖整个 M×N。

```cpp
for (int i = 0; i < 5; ++i) {
    gemmCpAsyncDoubleBuffered<<<grid, block>>>(...);
}
```

先执行 5 次预热，减少首次启动、缓存状态等因素对测量的影响。

```cpp
CUDA_CHECK(cudaDeviceSynchronize());
```

确保预热完成。

随后使用 CUDA event 包围 20 次 kernel：

```cpp
CUDA_CHECK(cudaEventRecord(start));

for (int i = 0; i < repeat; ++i) {
    gemmCpAsyncDoubleBuffered<<<grid, block>>>(...);
}

CUDA_CHECK(cudaEventRecord(stop));
CUDA_CHECK(cudaEventSynchronize(stop));
```

最后计算单次平均时间：

```cpp
return totalMs / repeat;
```

## 19. 编译方法

`cp.async` 必须生成 `sm_80+` 代码。当前工程的兼容构建命令是：

```bash
nvcc -O3 -std=c++17 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_80,code=compute_80 \
  gemm.cu -o gemm
```

含义：

```text
sm_75：为 Turing 保留同步回退版本
sm_80：为 Ampere 生成包含 cp.async 的机器码
compute_80：保留 PTX，让更新架构可以由驱动 JIT
```

只针对 Ampere 编译也可以使用：

```bash
nvcc -O3 -std=c++17 -arch=sm_80 gemm.cu -o gemm
```

检查 PTX 是否真的出现 `cp.async`：

```bash
nvcc -O3 -std=c++17 -arch=compute_80 -ptx gemm.cu -o gemm.ptx
rg "cp\.async" gemm.ptx
```

应该看到：

```text
cp.async.cg.shared.global
cp.async.commit_group
cp.async.wait_group 0
```

## 20. 当前编译资源占用

当前 `sm_80` 编译结果：

```text
64 registers/thread
16384 bytes shared memory/block
0 bytes spill stores
0 bytes spill loads
```

双缓冲增加了 shared-memory 占用，但 `cp.async` 避免像 V4 那样显式使用 `nextA`、`nextB` 普通寄存器保存下一块向量。

## 21. 常见误解

### `commit_group` 是否表示复制完成？

不是。它只提交一个异步复制 group，复制可以继续在后台进行。

### `wait_group 0` 能否替代 `__syncthreads()`？

不能。`wait_group 0` 等待当前线程发出的异步复制，`__syncthreads()` 同步整个 block。

### V5 是否让矩阵乘法本身异步执行？

不是。异步的是 global→shared 数据复制。FMA 计算仍然正常执行。

### V5 是否使用 Tensor Core？

没有。V5 仍然使用普通 FP32 CUDA Core 执行标量 FMA。Tensor Core 版本需要把内部 `4×4` 外积替换成 warp 级 `mma`/WMMA 矩阵乘加，并通常需要 FP16、BF16 或 TF32 输入。

### 尾块是否仍然正确？

正确。不满足 16 Byte 向量复制条件的线程会走 `loadFloat4OrZero` 同步回退，越界位置补 0。

## 22. 一句话总结

V5 的核心是：

```text
先用 cp.async 把 next tile 直接复制到备用 shared buffer，
不等待就计算 current tile，
到 buffer 切换点才 wait + __syncthreads。
```
