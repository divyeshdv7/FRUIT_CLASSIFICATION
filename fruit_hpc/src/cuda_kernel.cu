
#include <cuda_runtime.h>
#include <iostream>

__global__ void normalize_kernel(
    const unsigned char* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] / 255.0f;
    }
}

extern "C" void normalize_on_gpu(
    const unsigned char* input,
    float* output,
    int size
) {
    unsigned char* d_input;
    float* d_output;

    cudaMalloc(&d_input, size * sizeof(unsigned char));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(
        d_input, input,
        size * sizeof(unsigned char),
        cudaMemcpyHostToDevice
    );

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    normalize_kernel<<<blocks, threads>>>(
        d_input, d_output, size
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel error: "
                  << cudaGetErrorString(error) << std::endl;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(
        output, d_output,
        size * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_input);
    cudaFree(d_output);
}
