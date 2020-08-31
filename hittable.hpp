#pragma once
#include <stdio.h>
#include "ray.hpp"

// class material;

struct hit_record {
    float t;
    point3 p;
    vec3 normal;
    bool front_face;
    // material *matPtr;

    __host__ __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
};

class hittable {
public:
    __host__ __device__ virtual bool hit (const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};