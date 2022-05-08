#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>
#include <thread>
#include <chrono>

using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::istream;
using std::string;
using std::vector;
using std::tuple;
using std::ios_base;
using std::thread;
using std::chrono::steady_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;


#define MAX_CLASS_COUNT 32
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


class ProgramConfiguration {
private:
    string inputFilePath, outputFilePath;
    vector<vector<tuple<int, int>>> samples;
public:
    ProgramConfiguration(
            string inputFilePath,
            string outputFilePath,
            vector<vector<tuple<int, int>>> samples
    ) {
        this->inputFilePath = std::move(inputFilePath);
        this->outputFilePath = std::move(outputFilePath);
        this->samples = std::move(samples);
    }

    const string &getInputFilePath() const {
        return inputFilePath;
    }

    const string &getOutputFilePath() const {
        return outputFilePath;
    }

    const vector<vector<tuple<int, int>>> &getSamples() const {
        return samples;
    }

    unsigned long long getClassCount() const {
        return this->samples.size();
    }
};


struct Size2D {
    int width, height;

    int getSize() const {
        return width * height;
    }
};


ProgramConfiguration readProgramConfiguration(istream &inputStream) {
    string inputFilePath, outputFilePath;
    int classCount;
    vector<vector<tuple<int, int>>> samples;

    inputStream >> inputFilePath;
    inputStream >> outputFilePath;
    inputStream >> classCount;


    if (classCount > MAX_CLASS_COUNT) {
        cerr << "Got " << classCount << "classes, when max class count is " << MAX_CLASS_COUNT << "."
             << endl;
        exit(1);
    }

    for (int i = 0; i < classCount; ++i) {
        int pixelCount;
        inputStream >> pixelCount;
        vector<tuple<int, int>> pixels;

        for (int j = 0; j < pixelCount; ++j) {
            int x, y;
            inputStream >> x >> y;
            pixels.emplace_back(x, y);
        }

        samples.push_back(pixels);
    }

    return {inputFilePath, outputFilePath, samples};
}


ProgramConfiguration readProgramConfiguration(int argc, char *argv[]) {
    if (argc == 2) {
        auto configurationFilePath = argv[1];

        ifstream inputFile(configurationFilePath);
        if (!inputFile.is_open()) {
            cerr << "Could not open configuration file \"" << configurationFilePath << "\"" << endl;
            exit(1);
        }

        auto configuration = readProgramConfiguration(inputFile);
        inputFile.close();

        return configuration;
    }

    return readProgramConfiguration(cin);
}


tuple<Size2D, uchar4 *> readInputData(const ProgramConfiguration &configuration) {
    ifstream input(configuration.getInputFilePath(), ios_base::binary);

    if (!input.is_open()) {
        cerr << "Could not open input file \"" << configuration.getInputFilePath() << "\"." << endl;
        exit(1);
    }

    Size2D sourceSize{};
    input.read((char *) &sourceSize.width, sizeof(sourceSize.width));
    input.read((char *) &sourceSize.height, sizeof(sourceSize.height));

    auto *data = new uchar4[sourceSize.getSize()];
    input.read((char *) data, (long long) sizeof(data[0]) * sourceSize.getSize());

    input.close();

    return tuple<Size2D, uchar4 *>{sourceSize, data};
}


vector<float3> calculateClassesAverageValues(
        const ProgramConfiguration &configuration,
        uchar4 *data,
        Size2D dataSize
) {
    vector<float3> classesAverageValues;
    for (const auto &classSamples: configuration.getSamples()) {
        float r = 0, g = 0, b = 0;

        for (const auto &samplePixelCoord: classSamples) {
            auto x = std::get<0>(samplePixelCoord);
            auto y = std::get<1>(samplePixelCoord);

            auto pixel = data[y * dataSize.width + x];
            r += (float) pixel.x;
            g += (float) pixel.y;
            b += (float) pixel.z;
        }

        classesAverageValues.push_back(float3{
                (r / (float) classSamples.size()),
                (g / (float) classSamples.size()),
                (b / (float) classSamples.size())
        });
    }

    return classesAverageValues;
}


__constant__ int cudaClassCount;
__constant__ float3 cudaClassesAverageValues[MAX_CLASS_COUNT];


__global__ void kernel(uchar4 *data, Size2D dataSize, char *results) {
    auto totalThreadsCountX = gridDim.x * blockDim.x;
    auto totalThreadsCountY = gridDim.y * blockDim.y;

    auto threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    auto threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;

    for (auto x = threadIndexX; x < dataSize.width; x += totalThreadsCountX) {
        for (auto y = threadIndexY; y < dataSize.height; y += totalThreadsCountY) {
            auto p = data[y * dataSize.width + x];

            int bestClassIndex = -1;
            float bestClassSum = 0;
            for (auto i = 0; i < cudaClassCount; ++i) {
                auto avg = cudaClassesAverageValues[i];

                auto sum = powf((float) p.x - avg.x, 2)
                           + powf((float) p.y - avg.y, 2)
                           + powf((float) p.z - avg.z, 2);

                if (sum < bestClassSum || bestClassIndex == -1) {
                    bestClassSum = sum;
                    bestClassIndex = i;
                }
            }

            results[y * dataSize.width + x] = (char) bestClassIndex;
        }
    }
}


void cpuKernel(
        const vector<float3> &classesAverageValues,
        uchar4 *data, Size2D dataSize, char *results,
        int threadIndex, int threadCount
) {
    for (auto x = threadIndex; x < dataSize.width; x += threadCount) {
        for (auto y = threadIndex; y < dataSize.height; y += threadCount) {
            auto p = data[y * dataSize.width + x];

            int bestClassIndex = -1;
            float bestClassSum = 0;
            for (auto i = 0; i < cudaClassCount; ++i) {
                auto avg = cudaClassesAverageValues[i];

                auto sum = powf((float) p.x - avg.x, 2)
                           + powf((float) p.y - avg.y, 2)
                           + powf((float) p.z - avg.z, 2);

                if (sum < bestClassSum || bestClassIndex == -1) {
                    bestClassSum = sum;
                    bestClassIndex = i;
                }
            }

            results[y * dataSize.width + x] = (char) bestClassIndex;
        }
    }
}


void applyResultsToData(uchar4 *data, Size2D dataSize, const char *results) {
    for (auto i = 0; i < dataSize.height; ++i) {
        for (auto j = 0; j < dataSize.width; ++j) {
            auto index = i * dataSize.width + j;
            data[index].w = results[index];
        }
    }
}


void saveOutput(const ProgramConfiguration &configuration, uchar4 *data, Size2D dataSize) {
    ofstream output(configuration.getOutputFilePath(), ios_base::binary);
    if (!output.is_open()) {
        cerr << "Could not open output file \"" << configuration.getOutputFilePath() << "\"." << endl;
        exit(1);
    }

    output.write((char *) &dataSize.width, sizeof(dataSize.width));
    output.write((char *) &dataSize.height, sizeof(dataSize.height));

    output.write((char *) data, (long long) sizeof(uchar4) * dataSize.getSize());

    output.close();
}


long long testCpu(
        const vector<float3> &classesAverageValues,
        uchar4 *data, Size2D dataSize,
        int threadCount, int testCount
) {
    auto *results = new char[dataSize.getSize()];

    long long elapsedSumNs = 0;
    for (auto i = 0; i < testCount; ++i) {
        vector<thread> threads;
        auto start = steady_clock::now();

        for (int j = 0; j < threadCount; ++j) {
            threads.emplace_back(cpuKernel, classesAverageValues, data, dataSize, results, j, threadCount);
        }
        for (auto &th: threads) {
            th.join();
        }

        auto end = steady_clock::now();
        auto elapsedNs = duration_cast<nanoseconds>(end - start).count();

        elapsedSumNs += elapsedNs;
    }

    delete[] results;

    return elapsedSumNs / testCount;
}


float testGpu(
        const vector<float3> &classesAverageValues,
        uchar4 *data, Size2D dataSize,
        dim3 testGridDim, dim3 testBlockDim,
        int testCount
) {
    char *cudaResults;
    CSC(cudaMalloc(&cudaResults, sizeof(char) * dataSize.getSize()))

    float elapsedSumMs = 0;
    for (auto i = 0; i < testCount; ++i) {
        cudaEvent_t startEvent, endEvent;
        CSC(cudaEventCreate(&startEvent))
        CSC(cudaEventCreate(&endEvent))
        CSC(cudaEventRecord(startEvent))
        kernel<<<testGridDim, testBlockDim>>>(data, dataSize, cudaResults);
        CSC(cudaDeviceSynchronize())
        CSC(cudaGetLastError())
        CSC(cudaEventRecord(endEvent))
        CSC(cudaEventSynchronize(endEvent))

        float elapsedMs;
        CSC(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent))
        CSC(cudaEventDestroy(startEvent))
        CSC(cudaEventDestroy(endEvent))

        elapsedSumMs += elapsedMs;
    }

    CSC(cudaFree(cudaResults))

    return elapsedSumMs / (float) testCount;
}


void runTests(
        const ProgramConfiguration &configuration,
        const vector<float3> &classesAverageValues,
        uchar4 *data, Size2D dataSize
) {
    cout << "Input file path: \"" << configuration.getInputFilePath() << "\"" << endl;

    auto elapsedNs = testCpu(classesAverageValues, data, dataSize, 10, 10);
    cout << "CPU Elapsed time: " << (double) elapsedNs / 1000000 << "ms" << endl;

    tuple<dim3, dim3> testDimConfigurations[] = {
            {dim3(1, 1),   dim3(32, 1)},
            {dim3(1, 2),   dim3(32, 1)},
            {dim3(1, 4),   dim3(32, 1)},
            {dim3(1, 8),   dim3(32, 1)},
            {dim3(1, 16),  dim3(32, 1)},
            {dim3(1, 32),  dim3(32, 1)},
            {dim3(1, 32),  dim3(32, 2)},
            {dim3(1, 32),  dim3(32, 4)},
            {dim3(1, 32),  dim3(32, 8)},
            {dim3(1, 32),  dim3(32, 16)},
            {dim3(1, 32),  dim3(32, 32)},
            {dim3(2, 32),  dim3(32, 32)},
            {dim3(4, 32),  dim3(32, 32)},
            {dim3(8, 32),  dim3(32, 32)},
            {dim3(16, 32), dim3(32, 32)},
            {dim3(32, 32), dim3(32, 32)},
    };

    cout << "GPU Elapsed times:" << endl;
    for (auto &dims: testDimConfigurations) {
        auto testGridDim = std::get<0>(dims);
        auto testBlockDim = std::get<1>(dims);

        auto elapsedMs = testGpu(classesAverageValues, data, dataSize, testGridDim, testBlockDim, 10);
        cout << "(" << testGridDim.x << ", " << testGridDim.y << ") x ("
             << testBlockDim.x << ", " << testBlockDim.y << "): "
             << elapsedMs << "ms" << endl;
    }
}


void runNormal(
        const ProgramConfiguration &configuration,
        uchar4 *data, Size2D dataSize,
        const vector<float3> &classesAverageValues
) {
    auto classCount = (int) configuration.getClassCount();
    CSC(cudaMemcpyToSymbol(
            cudaClassCount,
            &classCount,
            sizeof(int)
    ))
    // https://stackoverflow.com/questions/6485496/how-to-get-stdvector-pointer-to-the-raw-data
    CSC(cudaMemcpyToSymbol(
            cudaClassesAverageValues,
            &*classesAverageValues.begin(),
            sizeof(float3) * configuration.getClassCount()
    ))

    // to GPU
    uchar4 *cudaData;
    char *cudaResults;
    CSC(cudaMalloc(&cudaData, sizeof(uchar4) * dataSize.getSize()))
    CSC(cudaMemcpy(cudaData, data, sizeof(uchar4) * dataSize.getSize(), cudaMemcpyHostToDevice))
    CSC(cudaMalloc(&cudaResults, sizeof(char) * dataSize.getSize()))

    kernel<<<dim3(32, 32), dim3(32, 32)>>>(cudaData, dataSize, cudaResults);

    // from GPU
    auto *results = new char[dataSize.getSize()];
    CSC(cudaMemcpy(results, cudaResults, sizeof(char) * dataSize.getSize(), cudaMemcpyDeviceToHost))
    CSC(cudaFree(cudaData))
    CSC(cudaFree(cudaResults))

    applyResultsToData(data, dataSize, results);
    saveOutput(configuration, data, dataSize);

    delete[] data;
    delete[] results;
}


int main(int argc, char *argv[]) {
    auto configuration = readProgramConfiguration(argc, argv);

    Size2D dataSize{};
    uchar4 *data;
    std::tie(dataSize, data) = readInputData(configuration);

    auto classesAverageValues = calculateClassesAverageValues(configuration, data, dataSize);

    runTests(configuration, classesAverageValues, data, dataSize);
//    runNormal(configuration, data, dataSize, classesAverageValues);

    return 0;
}
