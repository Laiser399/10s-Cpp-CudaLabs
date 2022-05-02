#include <iostream>
#include <tuple>
#include <string>
#include <fstream>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <iomanip>


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


__global__ void swapRowsKernel(double *matrix, int matrixSize, int i, int j, int startColumn) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto column = startColumn + threadIndex; column < matrixSize * 2; column += threadCount) {
        auto tm = matrix[column * matrixSize + i];
        matrix[column * matrixSize + i] = matrix[column * matrixSize + j];
        matrix[column * matrixSize + j] = tm;
    }
}


__global__ void nullifyRowsBelowKernel(double *matrix, int matrixSize, int diagonalIndex) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto row = diagonalIndex + 1 + threadIndex; row < matrixSize; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = diagonalIndex + 1; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


__global__ void nullifyRowsAboveKernel(double *matrix, int matrixSize, int diagonalIndex) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto row = threadIndex; row < diagonalIndex; row += threadCount) {
        auto multiplier = matrix[diagonalIndex * matrixSize + row] / matrix[diagonalIndex * matrixSize + diagonalIndex];
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + row] -= multiplier * matrix[column * matrixSize + diagonalIndex];
        }
    }
}


__global__ void normalizeDiagonalKernel(double *matrix, int matrixSize) {
    auto threadCount = gridDim.x * blockDim.x;
    auto threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    for (auto diagonalIndex = threadIndex; diagonalIndex < matrixSize; diagonalIndex += threadCount) {
        for (auto column = matrixSize; column < matrixSize * 2; ++column) {
            matrix[column * matrixSize + diagonalIndex] /= matrix[diagonalIndex * matrixSize + diagonalIndex];
        }
    }
}


void writeMatrix(ostream &output, double *matrix, int matrixSize) {
    for (auto row = 0; row < matrixSize; ++row) {
        for (auto column = 0; column < matrixSize; ++column) {
            output << matrix[column * matrixSize + row] << " ";
        }
        output << endl;
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

    double *cudaMatrix;
    cudaMalloc(&cudaMatrix, sizeof(double) * matrixSize * matrixSize * 2);
    cudaMemcpy(cudaMatrix, matrix, sizeof(double) * matrixSize * matrixSize * 2, cudaMemcpyHostToDevice);

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
        cudaMemcpy(
                &tm,
                cudaMatrix + diagonalIndex * matrixSize + maxElementRowIndex,
                sizeof(double),
                cudaMemcpyDeviceToHost
        );
        if (tm == 0) {
            cerr << "Could not calculate inverse matrix. Determinant of matrix equal zero." << endl;
            return 1;
        }

        if (diagonalIndex != maxElementRowIndex) {
            swapRowsKernel<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex, maxElementRowIndex, diagonalIndex);
        }

        nullifyRowsBelowKernel<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Back step
    for (auto diagonalIndex = matrixSize - 1; diagonalIndex > -1; --diagonalIndex) {
        nullifyRowsAboveKernel<<<1024, 1024>>>(cudaMatrix, matrixSize, diagonalIndex);
    }

    // Last step
    normalizeDiagonalKernel<<<1024, 1024>>>(cudaMatrix, matrixSize);

    cudaMemcpy(matrix, cudaMatrix, sizeof(double) * matrixSize * matrixSize * 2, cudaMemcpyDeviceToHost);
    cudaFree(cudaMatrix);

    cout << setprecision(10) << scientific;
    writeMatrix(cout, matrix + matrixSize * matrixSize, matrixSize);

    delete[] matrix;

    return 0;
}
