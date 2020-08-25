#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include "vec3.hpp"

#define CHECKCUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " " << func << "\n" ;
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(vec3 *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width)||(y >= height)) return;
    int pixelIdx = y * width + x;
    image[pixelIdx] = vec3(float(x) / width, float(y) / height, 0.8);
}

int main() {
    using namespace std;
    int width = 1200;
    int height = 600;
    int blockSize = 8;

    cout << "Image size " << width << " x " << height << " BlockSize " << blockSize << endl;

    size_t imageSize = width * height * 3 *sizeof(vec3);
    vec3 *image;
    CHECKCUDA(cudaMallocManaged((void **)&image, imageSize));

    dim3 blocks(width/blockSize + 1, height/blockSize + 1);
    dim3 threads(blockSize, blockSize);
    render<<<blocks, threads>>>(image, width, height);
    CHECKCUDA(cudaGetLastError());
    CHECKCUDA(cudaDeviceSynchronize());

    ofstream imgFile("img.ppm");
    imgFile << "P3\n" << width << " " << height << "\n255\n";

    for (int y = height-1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int pixelIdx = y * width + x;
            writeColor(imgFile, image[pixelIdx], 1);
        }
    }
    CHECKCUDA(cudaFree(image));
    return 0;
}
