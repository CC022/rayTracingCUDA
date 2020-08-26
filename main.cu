#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include "vec3.hpp"
#include "ray.hpp"

#define CHECKCUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " " << func << "\n" ;
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const point3 &center, float radius, const ray &r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius * radius;
    auto discriminant  = b*b - 4*a*c;
    return (discriminant > 0);
}

__device__ color ray_color(const ray &r) {
    if (hit_sphere(point3(0,0,-1), 0.5, r)) {
        return color(0, 1, 1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0,1.0,1.0) + t*color(0.5,0.7,1.0);
}

__global__ void render(vec3 *image, int width, int height, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width)||(y >= height)) return;
    int pixelIdx = y * width + x;
    ray r(origin, lowerLeftCorner + float(x)/float(width)*horizontal + float(y)/float(height)*vertical);
    image[pixelIdx] = ray_color(r);
}

int main() {
    using namespace std;
    clock_t start, stop;
    int width = 1200;
    int height = 600;
    int blockSize = 8;

    cout << "Image size " << width << " x " << height << " BlockSize " << blockSize << endl;

    //Camera
    auto aspectRatio = width / height;
    auto viewportHeight = 2.0;
    auto viewportWidth = aspectRatio * viewportHeight;
    auto focalLength = 1.0;

    point3 origin = point3(0,0,0);
    auto horizontal = vec3(viewportWidth, 0, 0);
    auto vertical = vec3(0, viewportHeight, 0);
    auto lowerLeftCorner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focalLength);

    size_t imageSize = width * height * 3 *sizeof(vec3);
    vec3 *image;
    CHECKCUDA(cudaMallocManaged((void **)&image, imageSize));

    dim3 blocks(width/blockSize + 1, height/blockSize + 1);
    dim3 threads(blockSize, blockSize);
    start = clock();
    render<<<blocks, threads>>>(image, width, height, lowerLeftCorner, horizontal, vertical, origin);
    CHECKCUDA(cudaGetLastError());
    CHECKCUDA(cudaDeviceSynchronize());
    stop = clock();
    cout << "kernal took " << (stop - start)/CLOCKS_PER_SEC << " seconds\n";

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
