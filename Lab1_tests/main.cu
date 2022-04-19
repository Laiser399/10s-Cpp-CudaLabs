#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <thread>

using std::cin;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using std::string;
using std::rand;
using std::vector;
using std::thread;

using std::chrono::steady_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

__global__ void kernel(const double *first, const double *second, unsigned int n, double *results) {
    auto totalThreadsCount = gridDim.x * blockDim.x;
    auto currentThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = currentThreadId; i < n; i += totalThreadsCount) {
        results[i] = first[i] > second[i]
                     ? first[i]
                     : second[i];
    }
}

void kernelCpu(const double *first, const double *second, unsigned int n, double *results, int threadsCount,
               int threadIndex) {
    for (int i = threadIndex; i < n; i += threadsCount) {
        results[i] = first[i] > second[i]
                     ? first[i]
                     : second[i];
    }
}

void calcSingleThread(const double *first, const double *second, unsigned int n, double *results) {
    for (int i = 0; i < n; ++i) {
        results[i] = first[i] > second[i]
                     ? first[i]
                     : second[i];
    }
}

void calcMultiThread(const double *first, const double *second, unsigned int n, double *results) {
    auto threadsCount = 10;
    vector<std::thread> threads;

    for (int i = 0; i < threadsCount; i++) {
        threads.emplace_back(kernelCpu, first, second, n, results, threadsCount, i);
    }

    for (auto &th: threads) {
        th.join();
    }
}

long long testCpuMultiThread(int n) {
    cout << "Start cpu test" << endl;

    auto *first = new double[n], *second = new double[n], *results = new double[n];
    for (int i = 0; i < n; ++i) {
        first[i] = rand();
        second[i] = rand();
    }
    cout << "Values init done" << endl;

    auto start = steady_clock::now();
    calcMultiThread(first, second, n, results);
    auto end = steady_clock::now();
    auto elapsedMs = duration_cast<milliseconds>(end - start).count();
    cout << "Multi thread calc done in " << elapsedMs << "ms" << endl;

    cout << "---Testing---" << endl;
    for (int i = 0; i < n; ++i) {
        if (results[i] < first[i] || results[i] < second[i]) {
            cout << "failed: " << first[i] << ", " << second[i] << " -> " << results[i] << endl;
        }
    }
    cout << "---Testing done---" << endl;

    return elapsedMs;
}

long long testCpuSingleThread(int n) {
    cout << "Start cpu test" << endl;

    auto *first = new double[n], *second = new double[n], *results = new double[n];
    for (int i = 0; i < n; ++i) {
        first[i] = rand();
        second[i] = rand();
    }
    cout << "Values init done" << endl;

    auto start = steady_clock::now();
    calcSingleThread(first, second, n, results);
    auto end = steady_clock::now();
    auto elapsedMs = duration_cast<milliseconds>(end - start).count();
    cout << "Single thread calc done in " << elapsedMs << "ms" << endl;

    cout << "---Testing---" << endl;
    for (int i = 0; i < n; ++i) {
        if (results[i] < first[i] || results[i] < second[i]) {
            cout << "failed: " << first[i] << ", " << second[i] << " -> " << results[i] << endl;
        }
    }
    cout << "---Testing done---" << endl;

    return elapsedMs;
}

float testGpu(int n) {

    cout << "Start cuda test" << endl;

    auto *first = new double[n], *second = new double[n];
    for (int i = 0; i < n; ++i) {
        first[i] = rand();
        second[i] = rand();
    }
    cout << "Values init done" << endl;

    double *cudaFirst;
    double *cudaSecond;
    double *cudaResults;
    cudaMalloc(&cudaFirst, sizeof(double) * n);
    cudaMalloc(&cudaSecond, sizeof(double) * n);
    cudaMalloc(&cudaResults, sizeof(double) * n);
    cudaMemcpy(cudaFirst, first, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSecond, second, sizeof(double) * n, cudaMemcpyHostToDevice);
    cout << "Cuda values init done" << endl;

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
    cudaEventRecord(startEvent);
    kernel<<<1024, 1024>>>(cudaFirst, cudaSecond, n, cudaResults);
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);
    cout << "Calc done in " << elapsedMs << "ms" << endl;

    cudaFree(cudaFirst);
    cudaFree(cudaSecond);

    auto *results = new double[n];
    cudaMemcpy(results, cudaResults, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(cudaResults);
    cout << "Results extracted" << endl;

    cout << "---Testing---" << endl;
    for (int i = 0; i < n; ++i) {
        if (results[i] < first[i] || results[i] < second[i]) {
            cout << "failed: " << first[i] << ", " << second[i] << " -> " << results[i] << endl;
        }
    }
    cout << "---Testing done---" << endl;

    delete[] first;
    delete[] second;
    delete[] results;

    return elapsedMs;
}

int main() {
    const int n = 1000000;

    cout << "Values count: " << n << endl << endl;

    auto testCount = 10;

    auto elapsedSum = 0.0;
    for (int i = 0; i < testCount; ++i) {
        cout << "---Test:" << i << "---" << endl;
        elapsedSum += testGpu(n);
        cout << endl;
    }
    cout << "Average elapsed time: " << elapsedSum / testCount << "ms" << endl;


    auto elapsedSingleThreadSum = 0.0;
    auto elapsedMultiThreadSum = 0.0;
    for (int i = 0; i < testCount; ++i) {
        cout << "---Test:" << i << "---" << endl;
        elapsedSingleThreadSum += (double) testCpuSingleThread(n);
        elapsedMultiThreadSum += (double) testCpuMultiThread(n);
        cout << endl;
    }
    cout << "Average elapsed time in single: " << elapsedSingleThreadSum / testCount << "ms" << endl;
    cout << "Average elapsed time in multi: " << elapsedMultiThreadSum / testCount << "ms" << endl;

    return 0;
}
