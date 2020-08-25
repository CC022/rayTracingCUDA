#include <iostream>
#include <fstream>
#include <time.h>

#define CHECKCUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

using namespace std;

void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " " << func << "\n" ;
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float *image, int max_x, int max_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= max_x)||(y >= max_y)) return;
    int pixelIdx = y*max_x*3 + x*3;
    image[pixelIdx] = float(x) / max_x;
    image[pixelIdx + 1] = float(y) / max_y;
    image[pixelIdx + 2] = 0.8;
}

int main() {
    int width = 1200;
    int height = 600;
    int blockSize = 8;

    cout << "Image size " << width << " x " << height << " BlockSize " << blockSize << endl;

    size_t imageSize = width * height * 3 *sizeof(float);
    float *image;
    CHECKCUDA(cudaMallocManaged((void **)&image, imageSize));

    dim3 blocks(width/blockSize + 1, height/blockSize + 1);
    dim3 threads(blockSize, blockSize);
    render<<<blocks, threads>>>(image, width, height);
    CHECKCUDA(cudaGetLastError());
    CHECKCUDA(cudaDeviceSynchronize());

    ofstream imgFile("img.ppm");
    imgFile << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height-1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int pixelIdx = j * width * 3 + i * 3;
            float r = image[pixelIdx];
            float g = image[pixelIdx + 1];
            float b = image[pixelIdx + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            imgFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    CHECKCUDA(cudaFree(image));
    return 0;
}
