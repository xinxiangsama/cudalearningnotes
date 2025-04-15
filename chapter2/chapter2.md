# Memory Architecture in GPUs

## Global Memory
*Global Memory* 是cuda编程中能够使用到的最大的内存，但也是最慢的内存。因此对其访问、写入需要有合适的策略。合并内存访问等策略（其中 warp 中的线程访问连续的内存位置）可以显著提高全局内存性能。

## Shared Memory
*Shared Memory* 有着高带宽低延迟，很适合线程之间共享数据。如果多个线程同时访问写入共享内存中同一位置处的数据会引发数据竞争。


# CUDA Execution Model

## Thread Hierarchy
核函数启动涉及到一个由线程块组成的网格，网格由两个层次组成：
- Thread Blocks : 网格的基本单元，线程块中包含多个线程，可以通过共享内存进行交互数据，相互之间可以同步进度。
- Threads : 最小的执行单元，块中的每个线程都有一个唯一的线程索引，可以使用 `threadIdx`、`blockIdx` 和 `blockDim` 等内置变量推导出索引。

```c++
__global__ void kernelFunction(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    // Kernel code to execute
}
```
其中`blockDim.x`表示线程块x方向上的维度，`blockIdx.x`表示线程块在该网格中水平方向的索引，`threadIdx.x`表示线程在该线程块中水平方向的索引。

## Grids
网格可以是一维、二维或三维的，从而为将复杂数据结构映射到 CUDA 线程提供灵活性。网格和线程块的尺度在核函数启动前指定：
```c++
dim3 gridSize(16, 16); // 16x16 grid  
dim3 blockSize(8, 8); // 8x8 threads per block  
kernelFunction<<<gridSize, blockSize>>>();
```

## Warp Execution
在block中的线程以 **32** 个为一组(*Wrap*)被执行，这是由GPU硬件管理器进行的，为了高效的并行执行。在同一个Wrap中线程在不同的数据上执行相同的指令。当 warp 中的线程采用不同的执行路径时，会发生发散，这可能会对性能产生不利影响。

## Synchronization
块中的线程可以使用显式同步语句（如` __syncthreads()` ）来同步其执行。这对于需要集体计算或通过共享内存共享中间结果的操作特别有用。

## Grid Synchronization
CUDA 中没有原生支持整个网格进行同步的功能。

# Global, Shared, and Constant Memory

# Streaming Multiprocessors(SMs)
流式多处理器 （SM） 是 NVIDIA GPU 中的核心计算单元，用于执行 CUDA 程序内核。每个 SM 都包含一个组 CUDA 核心，负责以大规模并行方式从多个线程执行单个指令。SM 的架构是了解 CUDA 编程的基础，因为它直接影响 GPU 应用程序的效率和性能。

每个 SM 都包含几个关键组件：CUDA 内核、特殊功能单元 （SFU）、加载/存储单元 （LSU） 和调度单元。CUDA 核心执行标准的算术和逻辑运算。SFU 处理更专业的指令，例如三角计算、指数和倒数。LSU 通过从 GPU 中的各个内存区域加载数据并将数据存储到这些区域来管理内存作。这些组件在调度单元的控制下和谐地协同工作，调度单元负责编排线程的执行。

单指令多线程的架构意味着，多个线程同时执行相同的指令。32个线程为一组被组织为Wrap(SM的执行单元)，当同一 warp 中的线程遵循不同的执行路径时，就会发生 warp 发散，导致 SM 序列化发散路径并可能降低性能。

每个 SM 都有多个 register 和特定数量的共享内存