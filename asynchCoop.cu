#include <cstdint>
#include <cstdio>
#include <thread>
#include <cuda_runtime.h>

#define Buffersize 1024

__device__ uint8_t* d_arg_buffer;
__device__ int flag = 1;

__device__ void test() {
    printf("test: d_arg_buffer[0] = %d\n", d_arg_buffer[0]);
}

__global__ void kernel() {
    printf("in kernel\n");
    printf("flag: %d\n", flag);

    while (atomicAdd(&flag, 0)) {
    }

    printf("flag down\n");
    test();    
}

void synchronizer(cudaStream_t stream, cudaEvent_t event) {
    printf("launched synchronizer\n");

    cudaStreamWaitEvent(stream, event, 0);

    int falseval = 0;
    cudaMemcpyToSymbolAsync(flag, &falseval, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	printf("cleared flag to 0 (async)\n");

    cudaStreamSynchronize(stream);

    printf("flag cleared\n");
}

int main() {
    int data = 42;

    cudaStream_t stream;
    cudaEvent_t event;
    cudaStreamCreate(&stream);
    cudaEventCreate(&event);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);

    uint8_t* dab;
    cudaMalloc(&dab, Buffersize);

    int one = 1;
    cudaMemcpyToSymbol(flag, &one, sizeof(int));
    cudaMemcpyToSymbol(d_arg_buffer, &dab, sizeof(dab));

    kernel<<<1, 1, 0, stream2>>>();

    cudaMemcpyAsync(dab, &data, 1, cudaMemcpyHostToDevice, stream);

    cudaEventRecord(event, stream);

    std::thread t1(synchronizer, std::ref(stream), std::ref(event));
    t1.join();

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    cudaFree(dab);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
	cudaStreamDestory(stream2);

    return 0;
}

