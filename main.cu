#include <iostream>
#include <fstream>
#include <float.h>
#include <chrono>
#include <cmath>
#include "vec3.hpp"
#include "ray.hpp"
#include "hittable.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"

#define CHECKCUDA(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA Error " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " " << func << "\n" ;
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray &r, hittable_list **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render(vec3 *image, int width, int height, vec3 lowerLeftCorner, vec3 horizontal, vec3 vertical, vec3 origin, 
    hittable_list **world) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width)||(y >= height)) return;
    int pixelIdx = y * width + x;
    ray r(origin, lowerLeftCorner + float(x)/float(width)*horizontal + float(y)/float(height)*vertical);
    image[pixelIdx] = ray_color(r, world);
}

__global__ void create_world(hittable **d_list, hittable_list **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hittable_list(d_list,2);
    }
}

__global__ void free_world(hittable **d_list, hittable_list **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main() {
    // Image
    using namespace std;
    int width = 1200;
    int height = 600;
    int blockSize = 8;
    cout << "Image size " << width << " x " << height << " BlockSize " << blockSize << endl;

    // World
    hittable **d_list;
    CHECKCUDA(cudaMallocManaged((void **)&d_list, 2*sizeof(hittable *)));
    hittable_list **d_world;
    CHECKCUDA(cudaMallocManaged((void **)&d_world, sizeof(hittable *)));
    create_world<<<1,1>>>(d_list,d_world);
    CHECKCUDA(cudaGetLastError());
    CHECKCUDA(cudaDeviceSynchronize());

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
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    render<<<blocks, threads>>>(image, width, height, lowerLeftCorner, horizontal, vertical, origin, d_world);
    CHECKCUDA(cudaGetLastError());
    CHECKCUDA(cudaDeviceSynchronize());
    chrono::steady_clock::time_point stop = chrono::steady_clock::now();
    cout << "Kernal took " << chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " ms\n";

    ofstream imgFile("img.ppm");
    imgFile << "P3\n" << width << " " << height << "\n255\n";

    for (int y = height-1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int pixelIdx = y * width + x;
            writeColor(imgFile, image[pixelIdx], 1);
        }
    }
    CHECKCUDA(cudaFree(image));
    free_world<<<1,1>>>(d_list,d_world);
    return 0;
}
