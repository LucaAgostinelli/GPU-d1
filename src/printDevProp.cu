#include <stdio.h>
#include <cuda_runtime.h>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);

    printf("  Memory Clock rate:           %.0f MHz\n",
           devProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:            %d bit\n",
           devProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:       %7.3f GB/s\n",
           2.0 * devProp.memoryClockRate *
           (devProp.memoryBusWidth / 8) / 1.0e6);

    printf("  Multiprocessors:             %3d\n",
           devProp.multiProcessorCount);

    printf("  Maximum number of threads per multiprocessor:  %d\n",
           devProp.maxThreadsPerMultiProcessor);

    printf("  Maximum number of threads per block:           %d\n",
           devProp.maxThreadsPerBlock);

    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           devProp.maxThreadsDim[0],
           devProp.maxThreadsDim[1],
           devProp.maxThreadsDim[2]);

    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           devProp.maxGridSize[0],
           devProp.maxGridSize[1],
           devProp.maxGridSize[2]);

    printf("  Total amount of shared memory per block:       %zu bytes\n",
           devProp.sharedMemPerBlock);
}

int main()
{
    int deviceCount = 0;

    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp devProp;

        cudaGetDeviceProperties(&devProp, i);

        printf("=== Device %d ===\n", i);
        printDevProp(devProp);
        printf("\n");
    }

    return 0;
}