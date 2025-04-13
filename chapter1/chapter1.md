# Introduction to CUDA Programming and C++

## overview of CUDA and GPU Computing

CUDA是由英伟达创建的并行计算平台和应用程序编程接口(API)模型。

CUDA编程模型主要包括：
- cpu端组织代码
- gpu端执行主要计算代码

下面让我们展示一个例子:
``` c++
#include<iostream>

__global__ void hellokernel(void){
    printf("Hello, World from GPU!\n");
}

int main(){
    // launch the kernel function
    hellokernel<<<1, 1>>>();
    // wait for the gpu to finish
    cudaDeviceSynchronize();
}
```

- **__global__** 关键词将此函数标记为在GPU端执行、CPU端调用的函数。
- **<<<1, 1>>>** 表示有一个块，每个块一个线程。
- `cudaDeviceSynchronize` 确保在cpu终止整个进程前，gpu完成自己的计算

NVIDIA GPU由流式多处理器组成(多个SM单元)，每个SM单元都可以同时管理多个线程(成千个)。
这些线程被组织为 *block*， *block*进一步被组织为 *grid*。每个 *block* 只被一个SM单元执行，在同一个 *block* 内的 *thread* 可以通过 *shared memory* 共享数据、同步运行进度。

CUDA 中的内存组成结构:
- **Global Memory** 对所有 *thread* 共享  ，有较高的延迟。
- **Shared Memory** 对于处于同一个 *block* 内的 *thread* 共享， 比 **Global Memory** 快很多。
- **Registers Memory** 每个 *thread* 独立的内存空间，很快但是很小。
- **Contant and Texture Memry** 针对不同访问模式优化的缓存内存。

GPU 计算的优势
- 大规模并行：GPU有成百上千个小核心用于同时处理多个任务。
- 高吞吐量： 由于其架构，GPU可以在单位时间处理大量数据。
- 能效比：？？？

## Why Use CUDA with C++?
**OOP** 原则支持封装、继承和多态，这对管理复杂的大型项目很重要。
``` c++
class Matrix
{
public:
    Matrix(const int& size){
        cudaMallocManaged(&data, N * N * sizeof(float));
    }

    virtual ~Martix(){
        cudaFree(data);
    }

    // Access
    float* getData() const{
        return data;
    }
    const int& getSize() const{
        return N;
    }
protected:
    int N;
    float* data;
}
```

**Extensive Ecosystem and Support**:

CUDA Toolkit提供了各种实用的程序，例如:
- `nvcc`(NVIDIA CUDA Compiler)
- `cuda-gdb`(CUDA Debugger)
- `cuBLAS` (CUDA Basic Linear Algebra Subprograms)
- `cuDNN` (CUDA Deep Neural Network library)

## Basic Concepts and Terminology
CUDA 的结构、执行模型、内存结构以及编程模型。
### CUDA Architecture
这是NVIDIA开发出来的为了让NVIDIA GPU用于更通用的用途的编程接口(API)。硬件层面上，一个GPU包含多个流式多处理器(SM,Streaming Multiprocessor)，每个SM包含成百上千个CUDA核心。

SM 是一个多线程处理器，可同时执行多个线程的指令。CUDA 核心是执行线程指令的各个处理器。

### Kernel
*kernel* 是一个用CUDA c++写的运行在GPU上的函数。不像一般在CPU上的函数，CUDA核函数会被N个不同的 *thread* 分别执行一次。每个执行这个核函数的 *thread* 都有一个独一的 `id`，方便我们标识处理不同的数据。

``` c++
__global__ void simpleKernel(int* d_array) 
{   
    int idx = threadIdx.x;   
    d_array[idx] *= 2;  
}
```

### Thread Hierarchy
- Thread : CUDA中最小的执行单元
- Block : 一组执行相同核函数并且可以通过 *shared memory* 共享数据的 *thread*
- Grid : 执行相同核函数的 *block* 的集合，整个网格块可以通过主机端启动

每个线程和线程块都有独一的id，用于决定操作哪些数据。*block* 被组织为一个 *grid*，可以是 1D, 2D 和 3D。

### Memory Hierarchy
- **Global Memory**：能够使用的最大的内存空间，可以被所有线程访问写入。其不在芯片上，，有着较高的延迟。
- **Shared Memory**：在芯片上的内，对于处于同一线程块中的线程均可访问，比 *global memory*的延迟低很多。
- **Local Memory**：对于每个线程是私有的，和 *global memory* 有着相同的延迟，也是在芯片外部。
- **Constant Memory**：储存常量的只读内存，在核函数执行期间不可以被改变。它具有高速缓存访问，如果数据访问模式表现出空间局部性，则提供比全局内存更快的访问。
- **Texture Memory**：缓存的只读内存，特别适用于空间定位的内存访问模式。它为特定作提供了优势，例如图形中的纹理映射和某些计算模式。

### Wrap
一个 *Wrap* 是一组32个线程，在同一个SM单元上被执行。在硬件层面上，每个 SM 在 warp 级别调度和执行指令。考虑 warp 的行为对性能至关重要，确保 warp 中的线程遵循相似的执行路径（即避免 warp 发散）。
> block 是软件层面的线程的组织， wrap是硬件层面的线程组织。
> 
> Block 是 CUDA 编程模型中的一个逻辑概念，用于组织线程。开发者在代码中定义线程如何分组到 block 中。
Warp 是硬件层面的概念，指的是 GPU 硬件中一次调度和执行的线程组，通常是 32 个线程。Warp 是由硬件自动管理的，开发者无法直接控制。
### Thread Divergence
线程发散是指在同一个Wrap中的线程进行了不同的执行路径(例如 if分支语句)。发散会导致性能降低，因为 warp 会串行执行 warp 中线程所采用的每个分支路径。
例如:
``` c++
__global__void divergence(int* d_data)
{
    int idx = threadIdx.x;
    if(idx % 2 == 0){
        d_dada[idx] *= 2;
    }else{
        //do nothing
    }
}
```
### Synchronization
同步是一个很重要的操作，在进行下一步操作时，我们有时需要确保上一步所有线程都已经完成了任务。在同一个线程块中，CUDA提供了 `__syncthreads()`去同步该线程块中的所有线程。

### Execution Configuration
当我们执行一个核函数的时候，需要去指定block和grid的维度，这是使用三尖括号语法完成的。
```c++
int numBlocks = 16;  
int threadsPerBlock = 256;  
simpleKernel<<<numBlocks, threadsPerBlock>>>(d_array);
```

### Stream