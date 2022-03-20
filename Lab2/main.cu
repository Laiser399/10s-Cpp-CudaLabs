#include <iostream>
#include <cstdio>

using std::fstream;
using std::ifstream;
using std::cout;
using std::endl;

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

int main() {
    size2D targetSize{};
    targetSize.width = 960;
    targetSize.height = 540;
    auto sourceFilePath = "..\\image1_binary";
    auto targetFilePath = "..\\image1_binary_res";

    size2D sourceSize{};

    // read source
    FILE *file = fopen(sourceFilePath, "rb");
    fread(&sourceSize.width, sizeof(int), 1, file);
    fread(&sourceSize.height, sizeof(int), 1, file);
    auto *data = new uchar4[sourceSize.getSize()];
    fread(data, sizeof(uchar4), sourceSize.getSize(), file);
    fclose(file);

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
    FILE *outFile = fopen(targetFilePath, "wb");
    fwrite(&targetSize.width, 1, sizeof(targetSize.height), outFile);
    fwrite(&targetSize.height, 1, sizeof(targetSize.height), outFile);
    fwrite(localResult, sizeof(localResult[0]), targetSize.getSize(), outFile);
    fclose(outFile);

    delete[] localResult;

    return 0;
}
