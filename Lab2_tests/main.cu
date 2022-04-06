#include <iostream>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>


using std::ifstream;
using std::ofstream;
using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::string;
using std::ios_base;
using std::thread;
using std::vector;
using std::chrono::steady_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;


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

struct size2D {
    int width, height;

    int getSize() const {
        return width * height;
    }
};

__global__ void kernel(cudaTextureObject_t source, uchar4 *target, size2D sourceSize, size2D targetSize) {
    auto totalThreadsCountX = gridDim.x * blockDim.x;
    auto totalThreadsCountY = gridDim.y * blockDim.y;
    auto threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
    auto threadIdY = blockDim.y * blockIdx.y + threadIdx.y;

    auto xCompressionCoefficient = sourceSize.width / targetSize.width;
    auto yCompressionCoefficient = sourceSize.height / targetSize.height;
    auto compressionBlockSize = xCompressionCoefficient * yCompressionCoefficient;

    for (unsigned int i = threadIdX; i < targetSize.width; i += totalThreadsCountX) {
        for (unsigned int j = threadIdY; j < targetSize.height; j += totalThreadsCountY) {

            auto rSum = 0, gSum = 0, bSum = 0, aSum = 0;
            for (int xShift = 0; xShift < xCompressionCoefficient; ++xShift) {
                for (int yShift = 0; yShift < yCompressionCoefficient; ++yShift) {
                    auto x = xCompressionCoefficient * i + xShift;
                    auto y = yCompressionCoefficient * j + yShift;
                    auto pixel = tex2D<uchar4>(source, (float) x, (float) y);
                    rSum += pixel.x;
                    gSum += pixel.y;
                    bSum += pixel.z;
                    aSum += pixel.w;
                }
            }

            target[j * targetSize.width + i] = make_uchar4(
                    rSum / compressionBlockSize,
                    gSum / compressionBlockSize,
                    bSum / compressionBlockSize,
                    aSum / compressionBlockSize);
        }
    }
}

bool isValidSizes(const size2D &sourceSize, const size2D &targetSize) {
    if (targetSize.width > sourceSize.width
        || targetSize.height > sourceSize.height
        || sourceSize.width % targetSize.width != 0
        || sourceSize.height % targetSize.height != 0) {
        return false;
    }

    return true;
}

float testGpu(const string &sourceFilePath, size2D targetSize,
              dim3 gridDimension, dim3 blockDimension) {
    // read source
    ifstream input(sourceFilePath, ios_base::binary);
    if (!input.is_open()) {
        cerr << "Could not open source file." << endl;
        exit(1);
    }

    size2D sourceSize{};
    input.read((char *) &sourceSize.width, sizeof(sourceSize.width));
    input.read((char *) &sourceSize.height, sizeof(sourceSize.height));

    auto *data = new uchar4[sourceSize.getSize()];
    input.read((char *) data, (long long) sizeof(data[0]) * sourceSize.getSize());
    input.close();

    // validate inputs
    if (!isValidSizes(sourceSize, targetSize)) {
        cerr << "Error: wrong target image size." << endl;
        exit(1);
    }

    // move data to device
    cudaArray *cudaData;
    auto channel = cudaCreateChannelDesc<uchar4>();
    auto sourcePitch = sizeof(uchar4) * sourceSize.width;
    CSC(cudaMallocArray(&cudaData, &channel, sourceSize.width, sourceSize.height))
    CSC(cudaMemcpy2DToArray(cudaData, 0, 0, data,
                            sourcePitch,
                            sizeof(uchar4) * sourceSize.width,
                            sourceSize.height,
                            cudaMemcpyHostToDevice))
    delete[] data;

    // creating texture
    cudaTextureObject_t tex;
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaData;
    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = false;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr))

    // allocating memory for result
    uchar4 *cudaResult;
    CSC(cudaMalloc(&cudaResult, sizeof(uchar4) * targetSize.getSize()))

    // core
    cudaEvent_t startEvent, endEvent;
    CSC(cudaEventCreate(&startEvent))
    CSC(cudaEventCreate(&endEvent))
    CSC(cudaEventRecord(startEvent))
    kernel<<<gridDimension, blockDimension>>>(tex, cudaResult, sourceSize, targetSize);
    CSC(cudaDeviceSynchronize())
    CSC(cudaGetLastError())
    CSC(cudaEventRecord(endEvent))
    CSC(cudaEventSynchronize(endEvent))

    // get elapsed time
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, startEvent, endEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(endEvent);

    // destroy
    CSC(cudaDestroyTextureObject(tex))
    CSC(cudaFreeArray(cudaData))
    CSC(cudaFree(cudaResult))

    return elapsedMs;
}

void kernelCpu(uchar4 *source, uchar4 *target, size2D sourceSize, size2D targetSize,
               int threadIndex, int threadsCount) {
    auto xCompressionCoefficient = sourceSize.width / targetSize.width;
    auto yCompressionCoefficient = sourceSize.height / targetSize.height;
    auto compressionBlockSize = xCompressionCoefficient * yCompressionCoefficient;

    for (int i = threadIndex; i < targetSize.getSize(); i += threadsCount) {
        auto tx = i / targetSize.width;
        auto ty = i % targetSize.width;

        auto rSum = 0, gSum = 0, bSum = 0, aSum = 0;
        for (int xShift = 0; xShift < xCompressionCoefficient; ++xShift) {
            for (int yShift = 0; yShift < yCompressionCoefficient; ++yShift) {
                auto sx = xCompressionCoefficient * tx + xShift;
                auto sy = yCompressionCoefficient * ty + yShift;
                auto pixel = source[sy * sourceSize.width + sx];
                rSum += pixel.x;
                gSum += pixel.y;
                bSum += pixel.z;
                aSum += pixel.w;
            }
        }

        target[i] = make_uchar4(
                rSum / compressionBlockSize,
                gSum / compressionBlockSize,
                bSum / compressionBlockSize,
                aSum / compressionBlockSize);
    }
}

long long testCpu(const string &sourceFilePath, size2D targetSize) {
    // read source
    ifstream input(sourceFilePath, ios_base::binary);
    if (!input.is_open()) {
        cerr << "Could not open source file." << endl;
        exit(1);
    }

    size2D sourceSize{};
    input.read((char *) &sourceSize.width, sizeof(sourceSize.width));
    input.read((char *) &sourceSize.height, sizeof(sourceSize.height));

    auto *source = new uchar4[sourceSize.getSize()];
    input.read((char *) source, (long long) sizeof(source[0]) * sourceSize.getSize());
    input.close();

    // validate inputs
    if (!isValidSizes(sourceSize, targetSize)) {
        cerr << "Error: wrong target image size." << endl;
        exit(1);
    }

    // core
    vector<thread> threads;
    auto threadCount = 10;
    auto *target = new uchar4[targetSize.getSize()];
    auto start = steady_clock::now();
    for (int i = 0; i < threadCount; ++i) {
        threads.emplace_back(kernelCpu, source, target, sourceSize, targetSize, i, threadCount);
    }
    for (auto &th: threads) {
        th.join();
    }
    auto end = steady_clock::now();
    auto elapsedNs = duration_cast<nanoseconds>(end - start).count();

    // destruction
    delete[] source;
    delete[] target;

    return elapsedNs;
}

int main(int argc, char *argv[]) {
    size2D targetSize{};
    string sourceFilePath;
    if (argc == 4) {
        sourceFilePath = argv[1];
        targetSize.width = std::stoi(argv[2]);
        targetSize.height = std::stoi(argv[3]);
    } else {
        cin >> sourceFilePath;
        cin >> targetSize.width >> targetSize.height;
    }

    const auto testsCount = 16;
    auto gridDimensions = new dim3[] {
        dim3(1, 1),

        dim3(1, 2),
        dim3(1, 4),
        dim3(1, 8),
        dim3(1, 16),
        dim3(1, 32),

        dim3(1, 32),
        dim3(1, 32),
        dim3(1, 32),
        dim3(1, 32),
        dim3(1, 32),

        dim3(2, 32),
        dim3(4, 32),
        dim3(8, 32),
        dim3(16, 32),
        dim3(32, 32),
    };
    auto blockDimensions = new dim3[] {
        dim3(32, 1),

        dim3(32, 1),
        dim3(32, 1),
        dim3(32, 1),
        dim3(32, 1),
        dim3(32, 1),

        dim3(32, 2),
        dim3(32, 4),
        dim3(32, 8),
        dim3(32, 16),
        dim3(32, 32),

        dim3(32, 32),
        dim3(32, 32),
        dim3(32, 32),
        dim3(32, 32),
        dim3(32, 32),
    };

    for (int i = 0; i < 16; ++i) {
        auto elapsedMs = testGpu(sourceFilePath, targetSize,
                                 gridDimensions[i], blockDimensions[i]);
        cout << "gridDim: " << gridDimensions[i].x << "x" << gridDimensions[i].y << "   "
             << "blockDim: " << blockDimensions[i].x << "x" << blockDimensions[i].y << "   "
             << "elapsed: " << elapsedMs << "ms" << endl;
    }


    auto elapsedNs = testCpu(sourceFilePath, targetSize);
    cout << "CPU Elapsed time: " << (double) elapsedNs / 1000000 << "ms" << endl;

    return 0;
}
