#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>

using std::ifstream;
using std::ofstream;
using std::cout;
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

    auto compressionCoefficientX = sourceSize.width / targetSize.width;
    auto compressionCoefficientY = sourceSize.height / targetSize.height;

    for (unsigned int i = threadIdX; i < targetSize.width; i += totalThreadsCountX) {
        for (unsigned int j = threadIdY; j < targetSize.height; j += totalThreadsCountY) {
            auto a = tex2D<uchar4>(source, compressionCoefficientX * i, compressionCoefficientY * j);
            auto b = tex2D<uchar4>(source, compressionCoefficientX * i + 1, compressionCoefficientY * j);
            auto c = tex2D<uchar4>(source, compressionCoefficientX * i, compressionCoefficientY * j + 1);
            auto d = tex2D<uchar4>(source, compressionCoefficientX * i + 1, compressionCoefficientY * j + 1);
            auto avgR = (a.x + b.x + c.x + d.x) / 4;
            auto avgG = (a.y + b.y + c.y + d.y) / 4;
            auto avgB = (a.z + b.z + c.z + d.z) / 4;
            auto avgA = (a.w + b.w + c.w + d.w) / 4;
            target[j * targetSize.width + i] = make_uchar4(avgR, avgG, avgB, avgA);
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
        cout << "Wrong count of arguments." << endl;
        return 1;
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
