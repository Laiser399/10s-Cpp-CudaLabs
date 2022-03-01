#include <iostream>
#include <iomanip>

using std::cin;
using std::cout;
using std::endl;
using std::setprecision;
using std::scientific;
using std::string;

__global__ void kernel(const double *first, const double *second, unsigned int n, double *results) {
    auto totalThreadsCount = gridDim.x * blockDim.x;
    auto currentThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = currentThreadId; i < n; i += totalThreadsCount) {
        results[i] = first[i] > second[i]
                     ? first[i]
                     : second[i];
    }
}

int main() {
    unsigned int n;
    cin >> n;

    auto *first = new double[n];
    auto *second = new double[n];
    for (int i = 0; i < n; ++i) {
        cin >> first[i];
    }
    for (int i = 0; i < n; ++i) {
        cin >> second[i];
    }

    double *cudaFirst;
    double *cudaSecond;
    cudaMalloc(&cudaFirst, sizeof(double) * n);
    cudaMalloc(&cudaSecond, sizeof(double) * n);
    cudaMemcpy(cudaFirst, first, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaSecond, second, sizeof(double) * n, cudaMemcpyHostToDevice);
    delete[] first;
    delete[] second;

    double *cudaResults;
    cudaMalloc(&cudaResults, sizeof(double) * n);
    kernel<<<1024, 1024>>>(cudaFirst, cudaSecond, n, cudaResults);
    cudaFree(cudaFirst);
    cudaFree(cudaSecond);

    auto *results = new double[n];
    cudaMemcpy(results, cudaResults, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(cudaResults);

    cout << setprecision(10) << scientific;
    for (unsigned int i = 0; i < n; ++i) {
        cout << results[i] << " ";
    }

    delete[] results;

    return 0;
}
