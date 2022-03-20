#include <iostream>
#include <fstream>
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

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *result, int sourceWidth, int sourceHeight, int width, int height) {
    auto totalThreadsCountX = gridDim.x * blockDim.x;
    auto totalThreadsCountY = gridDim.y * blockDim.y;
    auto threadIdX = blockDim.x * blockIdx.x + threadIdx.x;
    auto threadIdY = blockDim.y * blockIdx.y + threadIdx.y;

    auto compressionCoefficientX = sourceWidth / width;
    auto compressionCoefficientY = sourceHeight / height;

    for (unsigned int i = threadIdX; i < width; i += totalThreadsCountX) {
        for (unsigned int j = threadIdY; j < height; j += totalThreadsCountY) {
            auto a = tex2D(tex, compressionCoefficientX * i, compressionCoefficientY * j);
            auto b = tex2D(tex, compressionCoefficientX * i + 1, compressionCoefficientY * j);
            auto c = tex2D(tex, compressionCoefficientX * i, compressionCoefficientY * j + 1);
            auto d = tex2D(tex, compressionCoefficientX * i + 1, compressionCoefficientY * j + 1);
            auto avgR = (a.x + b.x + c.x + d.x) / 4;
            auto avgG = (a.y + b.y + c.y + d.y) / 4;
            auto avgB = (a.z + b.z + c.z + d.z) / 4;
            auto avgA = (a.w + b.w + c.w + d.w) / 4;
            result[j * width + i] = make_uchar4(avgR, avgG, avgB, avgA);
        }
    }
}


int main() {
    int targetWidth = 960, targetHeight = 540;
    auto sourceFilePath = "..\\image1_binary";
    auto targetFilePath = "..\\image1_binary_res";

    int sourceWidth, sourceHeight;

    FILE *file = fopen(sourceFilePath, "rb");
    fread(&sourceWidth, sizeof(int), 1, file);
    fread(&sourceHeight, sizeof(int), 1, file);
    auto *data = new uchar4[sourceWidth * sourceHeight];
    fread(data, sizeof(uchar4), sourceWidth * sourceHeight, file);
    fclose(file);

    if (targetWidth > sourceWidth
        || targetHeight > sourceHeight
        || sourceWidth % targetWidth != 0
        || sourceHeight % targetHeight != 0) {
        cout << "Error: wrong target image size." << endl;
        return 1;
    }

    cudaArray *cudaData;
    auto channel = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&cudaData, &channel, sourceWidth, sourceHeight))
//    CSC(cudaMemcpy2DToArray(cudaData, 0, 0, data,
//                            sourceWidth * sizeof(uchar4), sourceWidth * sizeof(uchar4), sourceHeight,
//                            cudaMemcpyHostToDevice))
    CSC(cudaMemcpyToArray(cudaData, 0, 0, data, sizeof(uchar4) * sourceWidth * sourceHeight, cudaMemcpyHostToDevice))
    delete[] data;

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.channelDesc = channel;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;
    CSC(cudaBindTextureToArray(tex, cudaData, channel))

    uchar4 *cudaResult;
    CSC(cudaMalloc(&cudaResult, sizeof(uchar4) * targetWidth * targetHeight))

    kernel<<<dim3(16, 16), dim3(32, 32)>>>(cudaResult, sourceWidth, sourceHeight, targetWidth, targetHeight);
    CSC(cudaDeviceSynchronize())
    CSC(cudaGetLastError())

    auto *localResult = new uchar4[targetWidth * targetHeight];
    CSC(cudaMemcpy(localResult, cudaResult, sizeof(localResult[0]) * targetWidth * targetHeight, cudaMemcpyDeviceToHost))
    CSC(cudaFreeArray(cudaData))
    CSC(cudaFree(cudaResult))
    CSC(cudaUnbindTexture(tex))

    FILE *outFile = fopen(targetFilePath, "wb");
    fwrite(&targetWidth, 1, sizeof(targetWidth), outFile);
    fwrite(&targetHeight, 1, sizeof(targetHeight), outFile);
    fwrite(localResult, sizeof(localResult[0]), targetWidth * targetHeight, outFile);
    fclose(outFile);

    delete[] localResult;

    return 0;
}
