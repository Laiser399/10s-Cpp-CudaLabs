#include <iostream>
#include <fstream>
#include <tuple>
#include <utility>
#include <vector>

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


vector<uchar4> calculateClassesAverageValues(const ProgramConfiguration &configuration, uchar4 *data, Size2D dataSize) {
    vector<uchar4> classesAverageValues;
    for (const auto &classSamples: configuration.getSamples()) {
        double r = 0, g = 0, b = 0;

        for (const auto &samplePixelCoord: classSamples) {
            auto x = std::get<0>(samplePixelCoord);
            auto y = std::get<1>(samplePixelCoord);

            auto pixel = data[y * dataSize.height + x];
            r += pixel.x;
            g += pixel.y;
            b += pixel.z;
        }

        classesAverageValues.push_back(uchar4{
                (unsigned char) (r / (double) classSamples.size()),
                (unsigned char) (g / (double) classSamples.size()),
                (unsigned char) (b / (double) classSamples.size()),
                (unsigned char) 0
        });
    }

    return classesAverageValues;
}


__constant__ int cudaClassCount;
__constant__ uchar4 cudaClassesAverageValues[MAX_CLASS_COUNT];


__global__ void kernel(uchar4 *data, Size2D dataSize, char *results) {
    auto totalThreadsCountX = gridDim.x * blockDim.x;
    auto totalThreadsCountY = gridDim.y * blockDim.y;

    auto threadIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    auto threadIndexY = blockIdx.y * blockDim.y + threadIdx.y;

    for (auto x = threadIndexX; x < dataSize.width; x += totalThreadsCountX) {
        for (auto y = threadIndexY; y < dataSize.height; y += totalThreadsCountY) {
            auto p = data[y * dataSize.width + x];

            auto bestClassIndex = cudaClassCount;
            auto bestClassSum = -1.0;
            for (auto i = 0; i < cudaClassCount; ++i) {
                auto avg = cudaClassesAverageValues[i];

                auto sum = powf((float) p.x - (float) avg.x, 2)
                           + powf((float) p.y - (float) avg.y, 2)
                           + powf((float) p.z - (float) avg.z, 2);

                if (sum < bestClassSum || bestClassSum < 0) {
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


int main(int argc, char *argv[]) {
    auto configuration = readProgramConfiguration(argc, argv);

    Size2D dataSize{};
    uchar4 *data;
    std::tie(dataSize, data) = readInputData(configuration);

    auto classesAverageValues = calculateClassesAverageValues(configuration, data, dataSize);

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
            sizeof(uchar4) * configuration.getClassCount()
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

    return 0;
}
