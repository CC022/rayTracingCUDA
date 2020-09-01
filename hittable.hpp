#pragma once
#include <stdio.h>
#include "ray.hpp"

// class material;

#include <stdio.h>
#include "ray.hpp"

// class material;

struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
    // material *matPtr;
};

class hittable {
public:
    __host__ __device__ virtual bool hit (const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};