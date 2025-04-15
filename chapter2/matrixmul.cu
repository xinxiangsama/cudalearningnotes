#include <cuda_runtime.h>
#include <iostream>
#include <array>

__global__ void matrixMul(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


int main(){
    const int N = 32;
    size_t size = N * N * sizeof(float);

    // 分配 host 内存（1D 数组表示 2D）
    // float* h_A = new float[N * N];
    // float* h_B = new float[N * N];
    // float* h_C = new float[N * N];

    std::array<float, N * N> h_A {};
    std::array<float, N * N> h_B {};
    std::array<float, N * N> h_C {};

    // 初始化 A 和 B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;  // 你可以用 rand() 或别的
        h_B[i] = 2.0f;
    }

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据到 device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // 设置 block/grid 维度
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);

    // 启动 kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待执行完成 + 检查错误
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;

    // 拷贝结果回 host
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * N; ++i) {
        std::cout << h_C[i]<<std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceProp prop;
    cudaGetDeviceProperties_v2(&prop, 0);
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

    return 0;
}
