#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>

using std::ifstream;
using std::ofstream;
using std::cout;
using std::cin;
using std::endl;
using std::string;
using std::ios_base;

#define CSC(call)                                                               \
    {                                                                           \
        auto error = call;                                                      \
        if (error != cudaSuccess) {                                             \
            cout << "Error " << cudaGetErrorName(error) << " in file \""        \
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
                    auto pixel = tex2D<uchar4>(source,(float) x,(float) y);
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

int main(int argc, char *argv[]) {
    size2D targetSize{};
    string sourceFilePath;
    string targetFilePath;
    if (argc == 5) {
        sourceFilePath = argv[1];
        targetFilePath = argv[2];
        targetSize.width = std::stoi(argv[3]);
        targetSize.height = std::stoi(argv[4]);
    } else {
        cin >> sourceFilePath;
        cin >> targetFilePath;
        cin >> targetSize.width >> targetSize.height;
    }

    // read source
    ifstream input(sourceFilePath, ios_base::binary);

    size2D sourceSize{};
    input.read((char *) &sourceSize.width, sizeof(sourceSize.width));
    input.read((char *) &sourceSize.height, sizeof(sourceSize.height));

    auto *data = new uchar4[sourceSize.getSize()];
    input.read((char *) data, (long long) sizeof(data[0]) * sourceSize.getSize());
    input.close();

    // validate inputs
    if (!isValidSizes(sourceSize, targetSize)) {
        cout << "Error: wrong target image size." << endl;
        return 1;
    }

    // move data to cuda memory
    cudaArray *cudaData;
    size_t pitch;
    CSC(cudaMallocPitch(&cudaData, &pitch, sourceSize.width * sizeof(uchar4), sourceSize.height))
    CSC(cudaMemcpy2D(cudaData, pitch, data,
                     sourceSize.width * sizeof(uchar4), sourceSize.width * sizeof(uchar4), sourceSize.height,
                     cudaMemcpyHostToDevice))
    delete[] data;

    // creating texture
    cudaTextureObject_t tex;
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = cudaData;
    resDesc.res.pitch2D.width = sourceSize.width;
    resDesc.res.pitch2D.height = sourceSize.height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc texDesc{};
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr))

    // allocating memory for result
    uchar4 *cudaResult;
    CSC(cudaMalloc(&cudaResult, sizeof(uchar4) * targetSize.getSize()))

    // core
    kernel<<<dim3(16, 16), dim3(32, 32)>>>(tex, cudaResult, sourceSize, targetSize);
    CSC(cudaDeviceSynchronize())
    CSC(cudaGetLastError())

    // move result from device to host
    auto *localResult = new uchar4[targetSize.getSize()];
    CSC(cudaMemcpy(localResult, cudaResult, sizeof(localResult[0]) * targetSize.getSize(),
                   cudaMemcpyDeviceToHost))
    CSC(cudaDestroyTextureObject(tex))
    CSC(cudaFree(cudaData))
    CSC(cudaFree(cudaResult))

    // save result
    ofstream output(targetFilePath, ios_base::binary);
    output.write((char *) &targetSize.width, sizeof(targetSize.width));
    output.write((char *) &targetSize.height, sizeof(targetSize.height));
    output.write((char *) localResult, (long long) sizeof(localResult[0]) * targetSize.getSize());
    output.close();

    delete[] localResult;

    return 0;
}
