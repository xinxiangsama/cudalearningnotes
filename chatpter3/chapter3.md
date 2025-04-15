# Understanding CUDA Kernels and Threads

本章介绍了 CUDA 内核和线程的基本知识，解释了什么是 CUDA 内核以及如何编写和启动它们。它讨论了线程索引、映射以及块和网格维度在并行任务执行中的作用。本章还讨论了使用共享内存的线程同步和协作，探讨了线程分歧问题，并重点介绍了设计高效 CUDA 内核的最佳实践。

## What is a CUDA Kernel
CUDA 核函数是用`c++`编写的，在GPU上运行的函数，可以被成千上万个线程同时运行。

在函数返回值前面加上`__global__`可以将这个函数声明明为核函数。
```c++
__global__ void myKernel(int *data, int value){
    // kernel code
}
```
核函数需要在主机端调用，指定其执行参数，例如 *Griddim*、*Blockdim*等。
```c++
dim3 gridDim(1);
dim3 blockDim(256);

myKernel<<<gridDim, blockDim>>>(d_data, value);
```

> 为何区分  cpu 和 gpu 端的的变量，通常在主机端变量名称前加上`h_`，设备端变量前面加上`d_`，分别表示 *host* 和 *device*。

在 kernel 函数中，每个线程都有一个唯一的索引，可以查询该索引以识别其在其 block 和 grid 中的位置。这些索引可以通过内置变量`threadIdex`、`blockIdx`、`blockDim`和`gridDim`来获取。
```c++
__global__ void myKernel(int *data, int value) {   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   
    data[idx] = value;
}
```

## Writing Your First CUDA Kernel