#include "utils.hpp"
#include <random>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("  Memory Clock rate:           %.0f Mhz\n", devProp.memoryClockRate * 1e-3f);

    printf("  Memory Bus Width:            %d bit\n", devProp.memoryBusWidth);

    printf("  Peak Memory Bandwidth:       %7.3f GB/s\n", 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth / 8) / 1.0e6);

    printf("  Multiprocessors:             %3d\n", devProp.multiProcessorCount);
    printf("  Maximum number of threads per multiprocessor:  %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n", devProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("  Total amount of shared memory per block:       %zu bytes\n", devProp.sharedMemPerBlock);
    return;
}

std::vector<float> generateRandomArray(int size)
{
    std::vector<float> arr(size);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < size; i++)
    {
        arr[i] = dist(gen);
    }

    return arr;
}

KernelResults::KernelResults(const std::string &kernel_name_, float avg_ms_, float min_ms_, float max_ms_, float variance_ms_, long long bytes_moved_)
    : kernel_name(kernel_name_), avg_ms(avg_ms_), min_ms(min_ms_), max_ms(max_ms_), variance_ms(variance_ms_), bytes_moved(bytes_moved_), eff_bw_gbs(0.0)
{
    if (avg_ms > 0.0f)
    {
        eff_bw_gbs = static_cast<double>(bytes_moved) / (avg_ms * 1e-3) / 1e9;
    }
}

void KernelResults::report_bandwidth(float peak_bw_gbs)
{
    double pct = 100.0 * this->eff_bw_gbs / peak_bw_gbs;
    printf("  eff BW: %.2f GB/s  (%.1f%% of peak %.0f GB/s)\n",
           this->eff_bw_gbs, pct, peak_bw_gbs);
}

void KernelResults::report_time()
{
    printf("  avg: %f ms\n", this->avg_ms);
    printf("  min: %f ms\n", this->min_ms);
    printf("  max: %f ms\n", this->max_ms);
}

void KernelResults::report(float peak_bw_gbs)
{
    printf("Kernel (%s):\n", this->kernel_name.c_str());
    this->report_bandwidth(peak_bw_gbs);
    printf("\n");
    this->report_time();
    printf("\n");
}