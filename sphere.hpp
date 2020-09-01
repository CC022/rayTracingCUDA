#pragma once
#include "hittable.hpp"
#include "vec3.hpp"

class sphere: public hittable {
public:
    vec3 center;
    float radius;
    // material *matPtr;
    
    __host__ __device__ sphere() {}
    __host__ __device__ sphere(vec3 cen, float r) : center(cen), radius(r) {};
    
    __host__ __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        vec3 oc = r.origin() - center;
        float a = dot(r.direction(), r.direction());
        float b = dot(oc, r.direction());
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - a*c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant))/a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                // rec.matPtr = matPtr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.at(rec.t);
                rec.normal = (rec.p - center) / radius;
                // rec.matPtr = matPtr;
                return true;
            }
        }
        return false;
    }
};