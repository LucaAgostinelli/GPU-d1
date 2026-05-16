#pragma once
#include <vector>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define TIMER_DEF struct timeval temp_1, temp_2

#define TIMER_START gettimeofday(&temp_1, (struct timezone *)0)

#define TIMER_STOP gettimeofday(&temp_2, (struct timezone *)0)

#define TIMER_ELAPSED ((temp_2.tv_sec - temp_1.tv_sec) + (temp_2.tv_usec - temp_1.tv_usec) / 1000000.0)

#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1);                                             \
        }                                                        \
    } while (0)

const int WARMUP_ITERATIONS = 20;
const int BENCHMARK_ITERATIONS = 100;

struct KernelResults
{
    std::string kernel_name;
    float avg_ms;
    float min_ms;
    float max_ms;
    float variance_ms;
    long long bytes_moved;
    double eff_bw_gbs;

    KernelResults(const std::string &kernel_name, float avg_ms_, float min_ms_, float max_ms_, float variance_ms_, long long bytes_moved_);

    void report_bandwidth(float peak_bw_gbs);
    void report_time();
    void report(float peak_bw_gbs);
};
void printDevProp(cudaDeviceProp devProp);

std::vector<float> generateRandomArray(int size);
