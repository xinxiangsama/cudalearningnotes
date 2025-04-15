#include <iostream>
#include <cuda_runtime.h>

__global__ void helloCUDA(){
    int idx = threadIdx.x;
    printf("Hello World from GPU! Thread ID: %d\n", idx);
}

int main(){
    helloCUDA<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}