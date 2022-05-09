#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <vector>
#include <chrono>


using std::cout;
using std::cerr;
using std::cin;
using std::endl;
using std::istream;
using std::ostream;
using std::ifstream;
using std::tuple;
using std::string;
using std::setprecision;
using std::scientific;
using std::thread;
using std::vector;
using std::function;
using std::chrono::steady_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;


#pragma region define CSC
#define CSC(call)                                                               \
    {                                                                           \
        auto error = call;                                                      \
        if (error != cudaSuccess) {                                             \
            cerr << "Error " << cudaGetErrorName(error) << " in file \""        \
                 << __FILE__ << "\", at line " << __LINE__ << ". "              \
                 << "Message: " << cudaGetErrorString(error) << endl;           \
            exit(1);                                                            \
        }                                                                       \
    }
#pragma endregion


struct doubleAbsComparator {
    __host__ __device__ bool operator()(double a, double b) {
        return abs(a) < abs(b);
    }
};


tuple<int, double *> readAndPrepareInput(istream &input) {
    int matrixSize;
    input >> matrixSize;

    auto *matrix = new double[matrixSize * matrixSize * 2];

    for (auto row = 0; row < matrixSize; ++row) {
        for (auto column = 0; column < matrixSize; ++column) {
            input >> matrix[column * matrixSize + row];
        }
    }

    auto *attachedMatrix = matrix + matrixSize * matrixSize;
    for (auto row = 0; row < matrixSize; ++row) {
        for (auto column = 0; column < matrixSize; ++column) {
            attachedMatrix[column * matrixSize + row] = row == column ? 1 : 0;
        }
    }

    return tuple<int, double *>{matrixSize, matrix};
}


tuple<int, double *> readAndPrepareInput(int argc, char *argv[]) {
    if (argc == 2) {
        auto inputFilePath = argv[1];

        ifstream input(inputFilePath);
        if (!input.is_open()) {
            cerr << "Could not open input file \"" << inputFilePath << "\"." << endl;
            exit(1);
        }

        auto result = readAndPrepareInput(input);

        input.close();

        return result;
    } else {
        return readAndPrepareInput(cin);
    }
}


#pragma region GPU kernels

__global__ void swapRows(double *matrix, int matrixSize, int i, int j, int startColumn) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto column = startColumn + threadIndex; column < matrixSize * 2; column += threadCount) {
        auto tm = matrix[column * matrixSize + i];
        matrix[column * matrixSize + i] = matrix[column * matrixSize + j];
        matrix[column * matrixSize + j] = tm;
    }
}


__global__ void nullifyRowsBelow(double *matrix, int matrixSize, int diagonalIndex) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto row = diagonalIndex + 1 + threadIndex; row < matrixSize; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = diagonalIndex + 1; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


__global__ void nullifyRowsAbove(double *matrix, int matrixSize, int diagonalIndex) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto row = threadIndex; row < diagonalIndex; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


__global__ void normalizeDiagonal(double *matrix, int matrixSize) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto diagonalIndex = threadIndex; diagonalIndex < matrixSize; diagonalIndex += threadCount) {
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + diagonalIndex] /= matrix[diagonalIndex * matrixSize + diagonalIndex];
        }
    }
}

#pragma endregion


#pragma region CPU kernels

void swapRows(
        double *matrix, int matrixSize, int i, int j, int startColumn,
        int threadIndex, int threadCount
) {
    for (auto column = startColumn + threadIndex; column < matrixSize * 2; column += threadCount) {
        auto tm = matrix[column * matrixSize + i];
        matrix[column * matrixSize + i] = matrix[column * matrixSize + j];
        matrix[column * matrixSize + j] = tm;
    }
}


void nullifyRowsBelow(
        double *matrix, int matrixSize, int diagonalIndex,
        int threadIndex, int threadCount
) {
    for (auto row = diagonalIndex + 1 + threadIndex; row < matrixSize; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = diagonalIndex + 1; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


void nullifyRowsAbove(
        double *matrix, int matrixSize, int diagonalIndex,
        int threadIndex, int threadCount
) {
    for (auto row = threadIndex; row < diagonalIndex; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


void normalizeDiagonal(
        double *matrix, int matrixSize,
        int threadIndex, int threadCount
) {
    for (auto diagonalIndex = threadIndex; diagonalIndex < matrixSize; diagonalIndex += threadCount) {
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + diagonalIndex] /= matrix[diagonalIndex * matrixSize + diagonalIndex];
        }
    }
}

#pragma endregion


void writeMatrix(ostream &output, double *matrix, int matrixSize) {
    for (auto row = 0; row < matrixSize; ++row) {
        for (auto column = 0; column < matrixSize; ++column) {
            output << matrix[column * matrixSize + row] << " ";
        }
        output << endl;
    }
}


#pragma region GPU tests

float testGpu(
        double *matrix, int matrixSize,
        dim3 testGridDim, dim3 testBlockDim
) {
    double *cudaMatrix;
    CSC(cudaMalloc(&cudaMatrix, sizeof(double) * matrixSize * matrixSize * 2))
    CSC(cudaMemcpy(cudaMatrix, matrix, sizeof(double) * matrixSize * matrixSize * 2, cudaMemcpyHostToDevice))

    cudaEvent_t startEvent, endEvent;
    CSC(cudaEventCreate(&startEvent))
    CSC(cudaEventCreate(&endEvent))
    CSC(cudaEventRecord(startEvent))

    // Forward step
    auto cudaMatrixDevicePtr = thrust::device_pointer_cast(cudaMatrix);
    for (auto diagonalIndex = 0; diagonalIndex < matrixSize; ++diagonalIndex) {
        auto columnPtr = cudaMatrixDevicePtr + diagonalIndex * matrixSize;
        auto maxElementPtr = thrust::max_element(
                columnPtr + diagonalIndex,
                columnPtr + matrixSize,
                doubleAbsComparator()
        );
        auto maxElementRowIndex = (int) (maxElementPtr - columnPtr);
        double tm;
        CSC(cudaMemcpy(
                &tm,
                cudaMatrix + diagonalIndex * matrixSize + maxElementRowIndex,
                sizeof(double),
                cudaMemcpyDeviceToHost
        ))
        if (tm == 0) {
            cerr << "Could not calculate inverse matrix. Determinant of matrix equal zero." << endl;
            exit(1);
        }

        if (diagonalIndex != maxElementRowIndex) {
            swapRows<<<testGridDim, testBlockDim>>>(
                    cudaMatrix, matrixSize, diagonalIndex, maxElementRowIndex, diagonalIndex);
        }

        nullifyRowsBelow<<<testGridDim, testBlockDim>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Back step
    for (auto diagonalIndex = matrixSize - 1; diagonalIndex > -1; --diagonalIndex) {
        nullifyRowsAbove<<<testGridDim, testBlockDim>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Last step
    normalizeDiagonal<<<testGridDim, testBlockDim>>>(cudaMatrix, matrixSize);

    CSC(cudaDeviceSynchronize())
    CSC(cudaGetLastError())
    CSC(cudaEventRecord(endEvent))
    CSC(cudaEventSynchronize(endEvent))

    float elapsedMs;
    CSC(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent))
    CSC(cudaEventDestroy(startEvent))
    CSC(cudaEventDestroy(endEvent))

    CSC(cudaFree(cudaMatrix))

    return elapsedMs;
}


float testGpu(double *matrix, int matrixSize,
              dim3 testGridDim, dim3 testBlockDim,
              int testCount
) {
    float elapsedSumMs = 0;
    for (auto i = 0; i < testCount; ++i) {
        elapsedSumMs += testGpu(matrix, matrixSize, testGridDim, testBlockDim);
    }
    return elapsedSumMs / (float) testCount;
}

#pragma endregion


#pragma region CPU tests

void parallelize(int threadCount, const function<void(int threadIndex)> &parallelizeFunc) {
    vector<thread> threads;
    for (auto i = 0; i < threadCount; ++i) {
        threads.emplace_back(parallelizeFunc, i);
    }
    for (auto &t: threads) {
        t.join();
    }
}


long long testCpu(double *matrix, int matrixSize, int threadCount) {
    auto *matrixCopy = new double[matrixSize * matrixSize * 2];
    std::copy(matrix, matrix + matrixSize * matrixSize * 2, matrixCopy);

    auto start = steady_clock::now();

    // Forward step
    for (auto diagonalIndex = 0; diagonalIndex < matrixSize; ++diagonalIndex) {
        auto columnPtr = matrixCopy + diagonalIndex * matrixSize;
        auto maxElementPtr = std::max_element(
                columnPtr + diagonalIndex,
                columnPtr + matrixSize,
                doubleAbsComparator()
        );
        if (*maxElementPtr == 0) {
            cerr << "Could not calculate inverse matrix. Determinant of matrix equal zero." << endl;
            exit(1);
        }

        auto maxElementRowIndex = (int) (maxElementPtr - columnPtr);
        if (diagonalIndex != maxElementRowIndex) {
            parallelize(
                    threadCount,
                    function<void(int)>([=](int threadIndex) {
                        swapRows(matrixCopy, matrixSize,
                                 diagonalIndex, maxElementRowIndex, diagonalIndex,
                                 threadIndex, threadCount);
                    })
            );
        }

        parallelize(
                threadCount,
                function<void(int)>([=](int threadIndex) {
                    nullifyRowsBelow(matrixCopy, matrixSize, diagonalIndex,
                                     threadIndex, threadCount);
                })
        );
    }

    // Back step
    for (auto diagonalIndex = matrixSize - 1; diagonalIndex > -1; --diagonalIndex) {
        parallelize(
                threadCount,
                function<void(int)>([=](int threadIndex) {
                    nullifyRowsAbove(matrixCopy, matrixSize, diagonalIndex,
                                     threadIndex, threadCount);
                })
        );
    }

    // Last step
    parallelize(
            threadCount,
            function<void(int)>([=](int threadIndex) {
                normalizeDiagonal(matrixCopy, matrixSize,
                                  threadIndex, threadCount);
            })
    );

    auto end = steady_clock::now();
    auto elapsedNs = duration_cast<nanoseconds>(end - start).count();

    delete[] matrixCopy;

    return elapsedNs;
}


long long testCpu(double *matrix, int matrixSize, int threadCount, int testCount) {
    long long elapsedSumNs = 0;
    for (auto i = 0; i < testCount; ++i) {
        elapsedSumNs += testCpu(matrix, matrixSize, threadCount);
    }
    return elapsedSumNs / testCount;
}

#pragma endregion


void runNormal(double *matrix, int matrixSize) {
    double *cudaMatrix;
    CSC(cudaMalloc(&cudaMatrix, sizeof(double) * matrixSize * matrixSize * 2))
    CSC(cudaMemcpy(cudaMatrix, matrix, sizeof(double) * matrixSize * matrixSize * 2, cudaMemcpyHostToDevice))

    // Forward step
    auto cudaMatrixDevicePtr = thrust::device_pointer_cast(cudaMatrix);
    for (auto diagonalIndex = 0; diagonalIndex < matrixSize; ++diagonalIndex) {
        auto columnPtr = cudaMatrixDevicePtr + diagonalIndex * matrixSize;
        auto maxElementPtr = thrust::max_element(
                columnPtr + diagonalIndex,
                columnPtr + matrixSize,
                doubleAbsComparator()
        );
        auto maxElementRowIndex = (int) (maxElementPtr - columnPtr);
        double tm;
        CSC(cudaMemcpy(
                &tm,
                cudaMatrix + diagonalIndex * matrixSize + maxElementRowIndex,
                sizeof(double),
                cudaMemcpyDeviceToHost
        ))
        if (tm == 0) {
            cerr << "Could not calculate inverse matrix. Determinant of matrix equal zero." << endl;
            exit(1);
        }

        if (diagonalIndex != maxElementRowIndex) {
            swapRows<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex, maxElementRowIndex, diagonalIndex);
        }

        nullifyRowsBelow<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Back step
    for (auto diagonalIndex = matrixSize - 1; diagonalIndex > -1; --diagonalIndex) {
        nullifyRowsAbove<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Last step
    normalizeDiagonal<<<1024, 1024>>>(cudaMatrix, matrixSize);

    CSC(cudaMemcpy(matrix, cudaMatrix, sizeof(double) * matrixSize * matrixSize * 2, cudaMemcpyDeviceToHost))
    CSC(cudaFree(cudaMatrix))

    cout << setprecision(10) << scientific;
    writeMatrix(cout, matrix + matrixSize * matrixSize, matrixSize);

    delete[] matrix;
}


void runTests(double *matrix, int matrixSize) {
    auto elapsedCpuNs = testCpu(matrix, matrixSize, 10, 10);
    cout << "CPU Elapsed time: " << (double) elapsedCpuNs / 1000000 << "ms" << endl;

    tuple<dim3, dim3> testDimConfigurations[] = {
            {dim3(1),    dim3(32)},
            {dim3(2),    dim3(32)},
            {dim3(4),    dim3(32)},
            {dim3(8),    dim3(32)},
            {dim3(16),   dim3(32)},
            {dim3(32),   dim3(32)},
            {dim3(32),   dim3(64)},
            {dim3(32),   dim3(128)},
            {dim3(32),   dim3(256)},
            {dim3(32),   dim3(512)},
            {dim3(32),   dim3(1024)},
            {dim3(64),   dim3(1024)},
            {dim3(128),  dim3(1024)},
            {dim3(256),  dim3(1024)},
            {dim3(512),  dim3(1024)},
            {dim3(1024), dim3(1024)},
    };

    cout << "GPU Elapsed times:" << endl;
    for (auto &dims: testDimConfigurations) {
        auto testGridDim = std::get<0>(dims);
        auto testBlockDim = std::get<1>(dims);

        auto elapsedMs = testGpu(matrix, matrixSize, testGridDim, testBlockDim, 10);
        cout << testGridDim.x << "x"
             << testBlockDim.x << ": "
             << elapsedMs << "ms" << endl;
    }
}


int main(int argc, char *argv[]) {
    /**
     * Важные нюансы в программе:
     *  Матрица хранится в транспонированном виде, т.е. не по строкам, а по столбцам.
     *  Матрица сразу создается вместе с присоединенной единичной матрицей, т.е. содержит N строк и 2N столбцов.
     */

    int matrixSize;
    double *matrix;
    std::tie(matrixSize, matrix) = readAndPrepareInput(argc, argv);

//    runNormal(matrix, matrixSize);
    runTests(matrix, matrixSize);
}
